import io
import torch

def should_do(i, doPer):
    return i % doPer == (doPer - 1)

def pil_2_bytes(pilimage):
    array = io.BytesIO()
    pilimage.save(array, format="PNG")
    return array.getvalue()

def cos_angle(t1, t2):
    # get the cosine of the angle between t1 and t2
    agreement = torch.dot(t1, t2)
    normalization = torch.norm(t1) * torch.norm(t2)
    if normalization == 0:
        return float(torch.equal(t1, t2))
    agreement /= normalization
    return agreement

def cos_angle_var(t1, t2):
    return cos_angle(t1.data, t2.data)
