import * as fs from 'fs';
import * as net from 'net';

export function readAllowedIps(filePath: string): string[] | undefined {
  if (!fs.existsSync(filePath)) {
    return undefined;
  }

  const cidrs = fs
    .readFileSync(filePath, 'utf-8')
    .split(/\r?\n/)
    .flatMap((line, index) => {
      const entry = line.replace(/#.*/, '').trim();
      if (!entry) {
        return [];
      }
      const [address, prefix, ...rest] = entry.split('/');
      const maxBits = net.isIPv6(address) ? 128 : 32;
      const bits = prefix === undefined ? maxBits : /^\d{1,3}$/.test(prefix) ? Number(prefix) : NaN;
      if (!net.isIP(address) || rest.length > 0 || !(bits <= maxBits)) {
        throw new Error(`${filePath}:${index + 1}: invalid IP address or CIDR "${entry}"`);
      }
      return [`${address}/${bits}`];
    });

  return cidrs.length > 0 ? cidrs : undefined;
}
