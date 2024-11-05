import torch
import torch.nn.functional as F

from src.losses import masked_anchor_losses, soft_cross_entropy
from src.models import CTModel, WSIModel


def main():
    torch.manual_seed(42)
    ct = torch.randn(4, 512); wsi = torch.randn(4, 12, 1792)
    labels = torch.tensor([0, 1, 0, 1]); targets = torch.randn(4, 3)
    targets[:, 1:] = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=torch.float32)
    masks = torch.tensor([[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.float32)
    weights = [torch.tensor(1.0), torch.tensor(1.0)]

    student = CTModel(with_anchors=True); teacher = WSIModel(with_anchors=True)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    teacher_logits, teacher_anchors, _ = teacher(wsi)
    student_logits, student_anchors, _ = student(ct)
    anchor_loss = sum(masked_anchor_losses(student_anchors, targets, masks, weights))
    soft_targets = torch.softmax(teacher_logits.detach(), dim=1)[:, 1]
    loss = F.cross_entropy(student_logits, labels) + 0.1 * anchor_loss + soft_cross_entropy(student_logits, soft_targets)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    deployment_logits, embedding = student.forward_diagnosis(ct)
    assert teacher_anchors.shape == student_anchors.shape == (4, 3)
    assert deployment_logits.shape == (4, 2) and embedding.shape == (4, 128)
    assert torch.isfinite(loss) and torch.isfinite(deployment_logits).all()
    print("smoke test passed: model forward, masked anchors, teacher/student step, CT-only inference")


if __name__ == "__main__": main()

