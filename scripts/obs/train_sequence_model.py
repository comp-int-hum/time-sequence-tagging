def evaluate(model, batches, metadata, classes, device):
    model.eval()
    num_correct = 0
    total = 0
    incorrect_texts = []
    
    input_data, input_labels = batches
    assert len(metadata) == len(input_data)
    assert len(metadata) == len(input_labels)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input, label, meta in zip(input_data, input_labels, metadata):
            # If empty batch, pass
            if input.size(0) == 0 or label.size(0) == 0:
                print("Empty batch")
                continue

            input = input.to(device)
            label = label.to(device)

            # Output and loss
            output = model(input, device = device)
            reshaped_output, reshaped_label = reshape_output_label(output, label, len(classes))

            
            # correct, curr_total, incorrect = calculate_accuracy(output, label, meta)
            
            preds = torch.argmax(reshaped_output, dim = 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(reshaped_label.cpu().tolist())
            # Get metrics
            num_correct += correct
            total += curr_total
            
            # Add incorrect texts
            incorrect_texts.extend(incorrect)
    
    # Calculate final metrics
    accuracy = num_correct / total
    print(f"All labels: {all_labels}")
    print(f"All preds: {all_preds}")
    cm = confusion_matrix(y_true = all_labels, y_pred = all_preds, labels = classes)
    f1 = 
    return accuracy, cm, incorrect_texts