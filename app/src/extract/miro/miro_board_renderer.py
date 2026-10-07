"""Render Miro board items to images with Pillow (no board export API required)."""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.utils.text_utils import clean_html_text

# Miro named fill colors (sticky notes) mapped to hex
STICKY_FILL_COLORS = {
    "gray": "#F5F6F8",
    "light_yellow": "#FFF9B1",
    "yellow": "#F5D128",
    "orange": "#FF9D48",
    "light_green": "#D5F692",
    "green": "#C9DF56",
    "dark_green": "#93D275",
    "cyan": "#67C6C0",
    "light_pink": "#FFCEE0",
    "pink": "#EA94BB",
    "violet": "#BE88C7",
    "red": "#F2555A",
    "light_blue": "#A6CCF5",
    "blue": "#6CD8FA",
    "dark_blue": "#9EA9FF",
    "black": "#000000",
}

DEFAULT_STICKY_FILL = "#FFF9B1"
DEFAULT_GEOMETRY = {
    "sticky_note": (228, 228),
    "card": (320, 88),
    "shape": (200, 120),
    "text": (200, 40),
    "diagram": (600, 400),
    "data_table_format": (600, 320),
    "doc_format": (784, 1105),
    "document": (960, 540),
    "image": (300, 300),
}

# Types rendered as a labeled placeholder box: the API exposes no drawable
# content for them, but their position still carries meaning.
PLACEHOLDER_LABELS = {
    "diagram": "Diagram",
    "data_table_format": "Table",
    "document": "Document",
    "image": "Image",
}
PLACEHOLDER_FILL = "#EFEFEC"

# Types that hold other items and therefore get an image of their own
CONTAINER_TYPES = {"frame", "diagram"}

FONT_SEARCH_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _resolve_font_path() -> Optional[Tuple[str, int]]:
    """Find a usable font file, preferring the JP face of Noto Sans CJK."""
    for path in FONT_SEARCH_PATHS:
        if not path or not os.path.exists(path):
            continue
        if path.endswith(".ttc"):
            for index in range(10):
                try:
                    font = ImageFont.truetype(path, 16, index=index)
                except OSError:
                    break
                if "JP" in font.getname()[0]:
                    return path, index
            return path, 0
        return path, 0
    return None


def attach_container_children(
    items: List[Dict[str, Any]], children_by_container: Dict[str, List[Dict[str, Any]]]
) -> int:
    """Fill in the parent of items nested in a diagram.

    The bulk items endpoint returns parent=null for diagram children, so the
    relationship is read back per container and passed in here.

    Args:
        items: Board items from the bulk items endpoint (updated in place)
        children_by_container: Diagram ID -> its children, from parent_item_id queries

    Returns:
        Number of items whose parent was filled in.
    """
    by_id = {item.get("id"): item for item in items}
    attached = 0
    for container_id, children in children_by_container.items():
        for child in children:
            target = by_id.get(child.get("id"))
            if target is not None and not (target.get("parent") or {}).get("id"):
                target["parent"] = {"id": container_id}
                attached += 1
    return attached


@dataclass
class RenderItem:
    """Board item with absolute canvas coordinates (center-based)."""

    item_id: str
    item_type: str
    x: float
    y: float
    width: float
    height: float
    content: str = ""
    fill: Optional[str] = None
    shape: str = "rectangle"
    parent_id: Optional[str] = None

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x - self.width / 2, self.y - self.height / 2, self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class RenderedRegion:
    """One rendered image covering a frame or a spatial cluster of items."""

    image: Image.Image
    region_type: str  # "frame" | "cluster"
    region_id: str
    title: str = ""
    item_ids: List[str] = field(default_factory=list)


class MiroBoardRenderer:
    """Render board items into per-frame / per-cluster images for vision LLM analysis.

    The goal is fidelity of layout and text, not visual fidelity: colored boxes
    with wrapped text preserve the spatial semantics of sticky-note boards.
    """

    RENDERABLE_TYPES = {
        "sticky_note",
        "card",
        "shape",
        "text",
        "image",
        "frame",
        "diagram",
        "data_table_format",
        "doc_format",
        "document",
    }

    def __init__(
        self,
        max_pixels: int = 2073600,
        margin: float = 60.0,
        cluster_gap: float = 300.0,
        max_scale: float = 2.0,
        min_scale: float = 0.05,
        background: str = "#FFFFFF",
    ):
        self.max_pixels = max_pixels
        self.margin = margin
        self.cluster_gap = cluster_gap
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.background = background
        self._font_spec = _resolve_font_path()
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        if self._font_spec is None:
            print("[MiroBoardRenderer] WARNING: no TrueType font found; text will use a low-quality bitmap font")

    # ---------- public API ----------

    def render(
        self, items: List[Dict[str, Any]], connectors: Optional[List[Dict[str, Any]]] = None
    ) -> List[RenderedRegion]:
        """Render raw Miro API items into one image per container / spatial cluster.

        Frames and diagrams are containers: each gets its own image showing the
        items nested inside it. Everything else is grouped by proximity.
        """
        parsed = self._parse_items(items)
        by_id = {i.item_id: i for i in parsed}
        containers = {i.item_id: i for i in parsed if i.item_type in CONTAINER_TYPES}

        regions: List[RenderedRegion] = []

        for container_id, container in containers.items():
            members = [i for i in parsed if i.item_id != container_id and self._is_descendant(i, container_id, by_id)]
            if not members:
                continue
            title = container.content
            if container.item_type == "diagram" and not title:
                title = f"[{PLACEHOLDER_LABELS['diagram']}]"
            # A diagram's own geometry can sit far from the items it holds, so
            # frame its region by the contents instead.
            bounds = container if container.item_type == "frame" else None
            regions.append(
                self._render_region(members, connectors, container.item_type, container_id, title, frame=bounds)
            )

        orphans = [i for i in parsed if i.item_type not in CONTAINER_TYPES and not self._container_of(i, by_id)]
        for cluster in self._cluster(orphans):
            regions.append(self._render_region(cluster, connectors, "cluster", _cluster_id(cluster), ""))

        return regions

    @staticmethod
    def _container_of(item: RenderItem, by_id: Dict[str, "RenderItem"]) -> Optional[str]:
        """Walk up the parent chain and return the nearest container's id."""
        seen = set()
        parent_id = item.parent_id
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_id.get(parent_id)
            if parent is None:
                return None
            if parent.item_type in CONTAINER_TYPES:
                return parent.item_id
            parent_id = parent.parent_id
        return None

    @staticmethod
    def _is_descendant(item: RenderItem, container_id: str, by_id: Dict[str, "RenderItem"]) -> bool:
        seen = set()
        parent_id = item.parent_id
        while parent_id and parent_id not in seen:
            if parent_id == container_id:
                return True
            seen.add(parent_id)
            parent = by_id.get(parent_id)
            if parent is None:
                return False
            parent_id = parent.parent_id
        return False

    # ---------- parsing ----------

    def _parse_items(self, items: List[Dict[str, Any]]) -> List[RenderItem]:
        by_id = {item.get("id"): item for item in items}
        parsed = []
        for item in items:
            item_type = item.get("type")
            if item_type not in self.RENDERABLE_TYPES:
                continue
            parsed_item = self._parse_item(item, by_id)
            if parsed_item:
                parsed.append(parsed_item)
        return parsed

    def _parse_item(self, item: Dict[str, Any], by_id: Dict[str, Any]) -> Optional[RenderItem]:
        item_type = item.get("type")
        data = item.get("data", {}) or {}
        style = item.get("style", {}) or {}
        geometry = item.get("geometry", {}) or {}

        default_w, default_h = DEFAULT_GEOMETRY.get(item_type, (200, 120))
        width = float(geometry.get("width") or default_w)
        height = float(geometry.get("height") or 0.0)

        if item_type == "card":
            content = clean_html_text(f"{data.get('title', '')}\n{data.get('description', '')}".strip())
        elif item_type == "frame":
            content = clean_html_text(data.get("title", "") or "")
        elif item_type == "doc_format":
            content = clean_html_text(data.get("content", "") or "")
        elif item_type in PLACEHOLDER_LABELS:
            title = clean_html_text(data.get("title", "") or "")
            label = PLACEHOLDER_LABELS[item_type]
            content = f"[{label}] {title}" if title else f"[{label}]"
        else:
            content = clean_html_text(data.get("content", "") or "")

        if not height:
            # Text boxes report width only and grow downwards, so derive the
            # height the wrapped content actually needs.
            if item_type == "text" and content:
                height = self._estimate_text_height(content, width, float(style.get("fontSize") or 14))
            else:
                height = default_h

        x, y = self._absolute_center(item, by_id)
        if item_type == "diagram":
            top_x, top_y = self._absolute_topleft(item, by_id)
            x, y = top_x + width / 2, top_y + height / 2

        fill = style.get("fillColor")
        if fill and not str(fill).startswith("#"):
            fill = STICKY_FILL_COLORS.get(fill, DEFAULT_STICKY_FILL)

        parent = item.get("parent") or {}

        return RenderItem(
            item_id=item.get("id", ""),
            item_type=item_type,
            x=x,
            y=y,
            width=width,
            height=height,
            content=content or "",
            fill=fill,
            shape=data.get("shape", "rectangle"),
            parent_id=parent.get("id"),
        )

    def _absolute_center(self, item: Dict[str, Any], by_id: Dict[str, Any]) -> Tuple[float, float]:
        """Resolve item center to absolute canvas coordinates.

        Two relative coordinate systems exist:
        - parent_top_left: the center is an offset from the parent's top-left.
        - diagram children: the API reports relativeTo=canvas_center, but the
          values are really local to the diagram's top-left corner.
        """
        position = item.get("position", {}) or {}
        x = float(position.get("x", 0.0))
        y = float(position.get("y", 0.0))

        parent = by_id.get((item.get("parent") or {}).get("id"))

        if parent is not None and parent.get("type") == "diagram":
            px, py = self._absolute_topleft(parent, by_id)
            return px + x, py + y

        if position.get("relativeTo") == "parent_top_left" and parent is not None:
            px, py = self._absolute_topleft(parent, by_id)
            return px + x, py + y

        return x, y

    def _estimate_text_height(self, text: str, width: float, font_size: float) -> float:
        """Height a text box needs once its content is wrapped at the given width."""
        size = max(6, int(round(font_size)))
        lines = self._wrap(text, self._get_font(size), max(width - 8.0, 8.0))
        return max(font_size, len(lines) * font_size * 1.4)

    def _absolute_topleft(self, item: Dict[str, Any], by_id: Dict[str, Any]) -> Tuple[float, float]:
        """Resolve an item's top-left corner in absolute canvas coordinates."""
        position = item.get("position", {}) or {}
        geometry = item.get("geometry", {}) or {}
        x = float(position.get("x", 0.0))
        y = float(position.get("y", 0.0))

        parent = by_id.get((item.get("parent") or {}).get("id"))

        # A nested diagram anchors by its top-left, unlike ordinary items
        if item.get("type") == "diagram" and position.get("relativeTo") == "parent_top_left" and parent is not None:
            px, py = self._absolute_topleft(parent, by_id)
            return px + x, py + y

        cx, cy = self._absolute_center(item, by_id)
        return cx - float(geometry.get("width") or 0.0) / 2, cy - float(geometry.get("height") or 0.0) / 2

    # ---------- clustering ----------

    def _cluster(self, items: List[RenderItem]) -> List[List[RenderItem]]:
        """Group orphan items whose gap-inflated bounding boxes touch (union-find)."""
        n = len(items)
        if n == 0:
            return []
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        half_gap = self.cluster_gap / 2
        boxes = []
        for item in items:
            x0, y0, x1, y1 = item.bbox
            boxes.append((x0 - half_gap, y0 - half_gap, x1 + half_gap, y1 + half_gap))

        for i in range(n):
            for j in range(i + 1, n):
                a, b = boxes[i], boxes[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    parent[find(i)] = find(j)

        clusters: Dict[int, List[RenderItem]] = {}
        for i, item in enumerate(items):
            clusters.setdefault(find(i), []).append(item)
        return list(clusters.values())

    # ---------- rendering ----------

    def _render_region(
        self,
        members: List[RenderItem],
        connectors: Optional[List[Dict[str, Any]]],
        region_type: str,
        region_id: str,
        title: str,
        frame: Optional[RenderItem] = None,
    ) -> RenderedRegion:
        if frame is not None:
            x0, y0, x1, y1 = frame.bbox
        else:
            x0 = min(i.bbox[0] for i in members)
            y0 = min(i.bbox[1] for i in members)
            x1 = max(i.bbox[2] for i in members)
            y1 = max(i.bbox[3] for i in members)

        x0 -= self.margin
        y0 -= self.margin
        x1 += self.margin
        y1 += self.margin
        title_band = 48.0 if title else 0.0

        world_w = max(x1 - x0, 1.0)
        world_h = max(y1 - y0 + title_band, 1.0)
        scale = math.sqrt(self.max_pixels / (world_w * world_h))
        scale = max(self.min_scale, min(self.max_scale, scale))

        img_w = max(int(world_w * scale), 32)
        img_h = max(int(world_h * scale), 32)
        image = Image.new("RGB", (img_w, img_h), self.background)
        draw = ImageDraw.Draw(image)

        def to_px(wx: float, wy: float) -> Tuple[float, float]:
            return (wx - x0) * scale, (wy - y0 + title_band) * scale

        if title:
            self._draw_text_in_box(draw, title, (0, 0, img_w, title_band * scale), align="left", bold_bg="#F0F0F0")

        member_ids = {m.item_id for m in members}

        # Larger items first so small notes stay on top
        boxes = {}
        for item in sorted(members, key=lambda i: i.width * i.height, reverse=True):
            bx0, by0 = to_px(item.bbox[0], item.bbox[1])
            bx1, by1 = to_px(item.bbox[2], item.bbox[3])
            boxes[item.item_id] = (bx0, by0, bx1, by1)
            self._draw_item(draw, item, (bx0, by0, bx1, by1))

        # Connectors last so arrowheads stay visible on top of the items
        if connectors:
            self._draw_connectors(draw, connectors, boxes, scale)

        return RenderedRegion(
            image=image, region_type=region_type, region_id=region_id, title=title, item_ids=sorted(member_ids)
        )

    def _draw_connectors(
        self,
        draw: ImageDraw.ImageDraw,
        connectors: List[Dict[str, Any]],
        boxes: Dict[str, Tuple[float, float, float, float]],
        scale: float,
    ) -> None:
        """Draw connectors as arrows between item borders."""
        for connector in connectors:
            start = (connector.get("startItem") or {}).get("id")
            end = (connector.get("endItem") or {}).get("id")
            if start not in boxes or end not in boxes:
                continue

            c0 = self._center_of(boxes[start])
            c1 = self._center_of(boxes[end])
            if c0 == c1:
                continue

            # Stop the line at each item's border instead of its center
            p0 = self._clip_to_box(c0, boxes[start], c1)
            p1 = self._clip_to_box(c1, boxes[end], c0)

            style = connector.get("style") or {}
            color = style.get("strokeColor") or "#4A4A4A"
            if not str(color).startswith("#"):
                color = "#4A4A4A"
            width = max(1, int(float(style.get("strokeWidth") or 2.0) * scale))

            draw.line([p0, p1], fill=color, width=width)

            head = max(5.0, min(14.0, 12.0 * scale))
            if str(style.get("endStrokeCap", "stealth")).lower() not in ("none", ""):
                self._draw_arrowhead(draw, p0, p1, head, color)
            if str(style.get("startStrokeCap", "none")).lower() not in ("none", ""):
                self._draw_arrowhead(draw, p1, p0, head, color)

            captions = connector.get("captions") or []
            label = clean_html_text((captions[0].get("content") if captions else "") or "")
            if label:
                mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
                font = self._get_font(max(10, int(14 * scale)))
                pad = 2
                tw = draw.textlength(label, font=font)
                th = font.size
                draw.rectangle(
                    (mid[0] - tw / 2 - pad, mid[1] - th / 2 - pad, mid[0] + tw / 2 + pad, mid[1] + th / 2 + pad),
                    fill=self.background,
                )
                draw.text(mid, label, fill="#333333", font=font, anchor="mm")

    @staticmethod
    def _center_of(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def _clip_to_box(
        center: Tuple[float, float],
        box: Tuple[float, float, float, float],
        toward: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Find where the ray from center toward a point exits the box."""
        dx = toward[0] - center[0]
        dy = toward[1] - center[1]
        half_w = max((box[2] - box[0]) / 2, 0.5)
        half_h = max((box[3] - box[1]) / 2, 0.5)

        t_values = []
        if abs(dx) > 1e-6:
            t_values.append(half_w / abs(dx))
        if abs(dy) > 1e-6:
            t_values.append(half_h / abs(dy))
        if not t_values:
            return center

        t = min(min(t_values), 1.0)
        return center[0] + dx * t, center[1] + dy * t

    @staticmethod
    def _draw_arrowhead(
        draw: ImageDraw.ImageDraw,
        tail: Tuple[float, float],
        tip: Tuple[float, float],
        size: float,
        color: str,
    ) -> None:
        dx, dy = tip[0] - tail[0], tip[1] - tail[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        base = (tip[0] - ux * size, tip[1] - uy * size)
        wing = size * 0.45
        draw.polygon(
            [
                tip,
                (base[0] + px * wing, base[1] + py * wing),
                (base[0] - px * wing, base[1] - py * wing),
            ],
            fill=color,
        )

    def _draw_item(self, draw: ImageDraw.ImageDraw, item: RenderItem, box: Tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = box
        if x1 - x0 < 2 or y1 - y0 < 2:
            return

        fill = item.fill
        text_color = "#1A1A1A"
        outline = "#8A8A8A"

        if item.item_type == "sticky_note":
            fill = fill or DEFAULT_STICKY_FILL
            outline = None
        elif item.item_type == "card":
            fill = fill or "#FFFFFF"
            outline = "#4262FF"
        elif item.item_type == "diagram":
            # A container drawn through its children; its own box carries an
            # unrelated position, so nothing is drawn for it here.
            return
        elif item.item_type == "doc_format":
            fill = fill or "#FFFFFF"
            outline = "#B0B0AA"
        elif item.item_type in PLACEHOLDER_LABELS:
            fill = PLACEHOLDER_FILL
            outline = "#B0B0AA"
        elif item.item_type == "text":
            fill = None
            outline = None

        if fill:
            if self._is_dark(fill):
                text_color = "#FFFFFF"
            radius = min(8.0, (x1 - x0) / 4, (y1 - y0) / 4)
            if item.item_type == "shape" and item.shape in ("circle", "ellipse", "oval"):
                draw.ellipse(box, fill=fill, outline=outline)
            elif item.item_type == "shape" and item.shape == "rhombus":
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=fill, outline=outline)
            else:
                draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)
        elif outline:
            draw.rounded_rectangle(box, radius=4, outline=outline)

        text = item.content
        if text:
            align = "left" if item.item_type in ("text", "card", "doc_format") else "center"
            self._draw_text_in_box(draw, text, box, align=align, color=text_color)

    def _draw_text_in_box(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        box: Tuple[float, float, float, float],
        align: str = "center",
        color: str = "#1A1A1A",
        bold_bg: Optional[str] = None,
    ) -> None:
        """Fit text into a box: wrap by measured width, shrink font until it fits."""
        x0, y0, x1, y1 = box
        box_w = x1 - x0 - 8
        box_h = y1 - y0 - 8
        if box_w < 4 or box_h < 4:
            return

        if bold_bg:
            draw.rectangle(box, fill=bold_bg)

        font_size = max(10, int(min(box_h, 40)))
        while font_size >= 8:
            font = self._get_font(font_size)
            lines = self._wrap(text, font, box_w)
            line_height = font_size * 1.25
            if len(lines) * line_height <= box_h or font_size == 8:
                break
            font_size -= 2

        font = self._get_font(font_size)
        lines = self._wrap(text, font, box_w)
        line_height = font_size * 1.25
        max_lines = max(1, int(box_h / line_height))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][: max(0, len(lines[-1]) - 1)] + "…"

        total_h = len(lines) * line_height
        ty = y0 + 4 + max(0.0, (box_h - total_h) / 2)
        for line in lines:
            if align == "center":
                tw = draw.textlength(line, font=font)
                tx = x0 + 4 + max(0.0, (box_w - tw) / 2)
            else:
                tx = x0 + 6
            draw.text((tx, ty), line, fill=color, font=font)
            ty += line_height

    def _wrap(self, text: str, font: ImageFont.ImageFont, max_width: float) -> List[str]:
        """Character-based wrapping (works for Japanese text without spaces)."""
        lines = []
        for raw_line in text.split("\n"):
            current = ""
            for char in raw_line:
                candidate = current + char
                if current and font.getlength(candidate) > max_width:
                    lines.append(current)
                    current = char
                else:
                    current = candidate
            lines.append(current)
        return [line for line in lines if line] or [""]

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        if size not in self._font_cache:
            if self._font_spec:
                path, index = self._font_spec
                self._font_cache[size] = ImageFont.truetype(path, size, index=index)
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    @staticmethod
    def _is_dark(hex_color: str) -> bool:
        try:
            value = hex_color.lstrip("#")
            r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError, IndexError:
            return False
        return (0.299 * r + 0.587 * g + 0.114 * b) < 100


def _cluster_id(cluster: List[RenderItem]) -> str:
    """Name a cluster after its lowest member id.

    The name has to survive across extractions: a positional index shifts as
    soon as an item is added or moved, which would orphan the knowledge already
    recorded under the old name.
    """
    return f"cluster_{min(item.item_id for item in cluster)}"


def group_items_by_region(items: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map every item id to the id of the region it is rendered in.

    Mirrors the grouping in MiroBoardRenderer.render() so that text chunks and
    rendered images share one file_id namespace: an item nested in a frame or
    diagram belongs to that container, and the rest fall into spatial clusters.

    Args:
        items: Raw Miro API items, with container children already attached

    Returns:
        Dictionary mapping item_id to region_id
    """
    renderer = MiroBoardRenderer()
    parsed = renderer._parse_items(items)
    by_id = {i.item_id: i for i in parsed}

    region_of: Dict[str, str] = {}
    orphans: List[RenderItem] = []

    for item in parsed:
        container_id = renderer._container_of(item, by_id)
        if container_id:
            region_of[item.item_id] = container_id
        elif item.item_type in CONTAINER_TYPES:
            # A top-level container is the region its own children belong to
            region_of[item.item_id] = item.item_id
        else:
            orphans.append(item)

    for cluster in renderer._cluster(orphans):
        region_id = _cluster_id(cluster)
        for item in cluster:
            region_of[item.item_id] = region_id

    return region_of


def main() -> None:
    """Render a board (URL or saved items JSON) to PNG files for visual inspection."""
    import argparse

    parser = argparse.ArgumentParser(description="Render Miro board items to PNG images")
    parser.add_argument("source", help="Miro board URL or path to a JSON file with raw items")
    parser.add_argument("--out", default="output/miro_render", help="Output directory for PNG files")
    parser.add_argument("--cluster-gap", type=float, default=300.0, help="Max gap (board units) within a cluster")
    args = parser.parse_args()

    if args.source.startswith("http") or args.source.startswith("miro:"):
        from src.extract.miro.miro_text_extractor import fetch_board, parse_board_id_from_url
        from src.utils.miro_client import MiroClient

        board_id = parse_board_id_from_url(args.source)
        print(f"Fetching items from board {board_id} ...")
        _, items, connectors, children_by_container = fetch_board(MiroClient(), board_id)
        attached = attach_container_children(items, children_by_container)
        print(f"Fetched {len(items)} items and {len(connectors)} connectors ({attached} nested in diagrams)")
    else:
        with open(args.source, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            items = loaded.get("items")
            connectors = loaded.get("connectors") or []
        else:
            items = loaded
            connectors = []

    renderer = MiroBoardRenderer(cluster_gap=args.cluster_gap)
    regions = renderer.render(items, connectors)

    os.makedirs(args.out, exist_ok=True)
    for region in regions:
        filename = f"{region.region_type}_{region.region_id}.png"
        path = os.path.join(args.out, filename)
        region.image.save(path)
        label = f" title={region.title!r}" if region.title else ""
        print(f"  {path}  {region.image.size[0]}x{region.image.size[1]}  items={len(region.item_ids)}{label}")
    print(f"Rendered {len(regions)} region(s) to {args.out}")


if __name__ == "__main__":
    main()
