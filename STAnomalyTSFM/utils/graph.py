import torch


def grid_to_graph(shape):
    h, w = shape
    edge = []
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            cur_point = i * w + j
            left_point = cur_point - 1
            right_point = cur_point + 1
            upper_point = cur_point - w
            lower_point = cur_point + w
            left_upper = upper_point - 1
            left_lower = lower_point - 1
            right_upper = upper_point + 1
            right_lower = lower_point + 1
            edge.append([cur_point, left_point])
            edge.append([cur_point, right_point])
            edge.append([cur_point, upper_point])
            edge.append([cur_point, lower_point])
            edge.append([cur_point, left_upper])
            edge.append([cur_point, right_upper])
            edge.append([cur_point, left_lower])
            edge.append([cur_point, right_lower])


    for j in range(1, w - 1):
        cur_point = 0 * w + j
        left_point = cur_point - 1
        right_point = cur_point + 1
        lower_point = cur_point + w
        left_lower = lower_point - 1
        right_lower = lower_point + 1
        edge.append([cur_point, left_point])
        edge.append([cur_point, right_point])
        edge.append([cur_point, lower_point])
        edge.append([cur_point, left_lower])
        edge.append([cur_point, right_lower])

        cur_point = (h - 1) * w + j
        left_point = cur_point - 1
        right_point = cur_point + 1
        upper_point = cur_point - w
        left_upper = upper_point - 1
        right_upper = upper_point + 1
        edge.append([cur_point, left_point])
        edge.append([cur_point, right_point])
        edge.append([cur_point, upper_point])
        edge.append([cur_point, left_upper])
        edge.append([cur_point, right_upper])


    for i in range(1, h - 1):
        cur_point = i * w
        right_point = cur_point + 1
        upper_point = cur_point - w
        lower_point = cur_point + w
        right_upper = upper_point + 1
        right_lower = lower_point + 1
        edge.append([cur_point, right_point])
        edge.append([cur_point, upper_point])
        edge.append([cur_point, lower_point])
        edge.append([cur_point, right_upper])
        edge.append([cur_point, right_lower])

        cur_point = i * w + (w - 1)
        left_point = cur_point - 1
        upper_point = cur_point - w
        lower_point = cur_point + w
        left_upper = upper_point - 1
        left_lower = lower_point - 1
        edge.append([cur_point, left_point])
        edge.append([cur_point, upper_point])
        edge.append([cur_point, lower_point])
        edge.append([cur_point, left_upper])
        edge.append([cur_point, left_lower])

    edge.append([0, 1])
    edge.append([0, w])
    edge.append([0, w + 1])

    edge.append([w - 1, w - 2])
    edge.append([w - 1, 2 * w - 1])
    edge.append([w - 1, 2 * w - 2])

    edge.append([w * (h - 1), w * (h - 1) + 1])
    edge.append([w * (h - 1), w * (h - 2)])
    edge.append([w * (h - 1), w * (h - 2) + 1])

    edge.append([w * h - 1, w * h - 2])
    edge.append([w * h - 1, w * (h - 1) - 1])
    edge.append([w * h - 1, w * (h - 1) - 2])

    return torch.tensor(edge, dtype=torch.long).T


def normalize(A):
    I = torch.eye(A.shape[0], device=A.device)
    A_hat = A + I
    D = torch.diag(1 / A_hat.sum(1).sqrt())
    return D @ A_hat @ A
