
# QKV_Proj, 2GPU
def k8192_n5120(m):
    if m < 17:
        return 16
    elif m < 49:
        return 8
    else:
        return 4
    
# O_Proj, 2GPU
def k4096_n8192(m):
    # if m <= 32:
    #     return 8
    # else:
    #    return 4
    return 8 if m < 33 else 4
    
# Up_Gate_Fused_Proj, 2GPU
def k8192_n28672(m):
    if m < 12:
        return 8
    elif m < 33:
        return 2
    elif m < 49:
        return 8
    else:
        return 1
    
# Down_Proj, 2GPU
def k14336_n8192(m):
    if m < 17:
        return 16
    elif m < 33:
        return 8
    else:
        return 4

def default(m):
    return 8

def get_strategy(k, n):
    if k == 8192 and n == 5120:
        return k8192_n5120
    elif k == 4096 and n == 8192:
        return k4096_n8192
    elif k == 8192 and n == 28672:
        return k8192_n28672
    elif k == 14336 and n == 8192:
        return k14336_n8192
    else:
        return default
