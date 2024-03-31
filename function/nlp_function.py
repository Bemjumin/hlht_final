
import torch
import time
def test(test_dataloader, model, device):
    model.eval()
    total = 0
    correct = 0
    model.to(device)
    print('------------start test:-------------')
    with torch.no_grad():
        for data in test_dataloader:
            tid, ttid, mask, labels = data
            tid, ttid, mask, labels = tid.to(device), ttid.to(device), mask.to(device), labels.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            logits = model(input_ids=tid, token_type_ids=ttid, attention_mask=mask).logits.argmax(dim=1)
            total += labels.size(0)
            correct += (logits == labels).sum()

    print('Accuracy of the network on the dataset is: %d %%' % (
            100 * correct / total))
    return correct / total