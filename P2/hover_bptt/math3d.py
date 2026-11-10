import torch
def quaternion(euler):
    roll, pitch, yaw = (euler/2).unbind(-1)
    cr, sr, cp, sp, cy, sy = roll.cos(), roll.sin(), pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin()
    return torch.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy), -1)

def multiply(a, b):
    s, v, t, w = a[:, :1], a[:, 1:], b[:, :1], b[:, 1:]
    return torch.cat((s*t-(v*w).sum(-1, keepdim=True), s*w+t*v+torch.linalg.cross(v, w)), -1)
