Merge method for two diffusion models.

ORBIT explicitly decomposes B into components parallel and orthogonal to A, nudges along A’s direction with a (optionally clipped) projection coefficient while injecting the orthogonal residual at a separate weight to preserve A’s structure.

Uses the [sd_mecha](https://github.com/ljleb/sd-mecha) api.