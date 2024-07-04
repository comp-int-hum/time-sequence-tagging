def unpack_data(datapoint):
    data = datapoint.pop("embeddings")
    paragraph_labels = datapoint["paragraph_labels"]
    chapter_labels = datapoint["chapter_labels"]
    metadata = datapoint

    return data, paragraph_labels, chapter_labels, metadata


def get_specified_batch(filepath, batch_size=32, device="cuda"):
    data_batches, paragraph_batches, chapter_batches, metadata_batches = [], [], []
    data_batch, paragraph_batch, chapter_batch, metadata_batch = [], [], [], []
    total_datapoints = 0
    positive_datapoints = 0
    
    # Helper function for appending batches and padding if model_type is sequence_tagger
    def append_data(batch_data, batch_paragraph, batch_chapter, batch_metadata):
        data_batch = rnn_utils.pad_sequence([torch.tensor(d) for d in batch_data], batch_first=True)
        paragraph_batch = rnn_utils.pad_sequence([torch.tensor(p) for p in batch_paragraph], batch_first=True)
        chapter_batch = rnn_utils.pad_sequence([torch.tensor(c) for c in batch_chapter], batch_first=True)
        data_batches.append(data_batch.to(device))
        paragraph_batches.append(paragraph_batch.to(device))
        chapter_batches.append(chapter_batch.to(device))
        metadata_batches.append(batch_metadata)
        
    # Open data file
    with gzip.open(filepath, "r") as gzipfile:
        datapoints = jsonlines.Reader(gzipfile)
        for datapoint in datapoints:
            total_datapoints += 1
            data, paragraph_labels, chapter_labels, metadata = unpack_data(datapoint)
                
            # Append data
            data_batch.append(data)
            paragraph_batch.append(paragraph_labels)
            chapter_batch.append(chapter_labels)
            metadata_batch.append(metadata)
            
            # Add batch if batch_sized has been reached
            if len(data_batch) == batch_size:
                append_data(data_batch, paragraph_batch, chapter_batch, metadata_batch)
                data_batch, paragraph_batch, chapter_batch, metadata_batch = [], [], [], []
        
        # Add leftover data items to a batch
        if data_batch:
            append_data(data_batch, paragraph_batch, chapter_batch, metadata_batch)
            data_batch, paragraph_batch, chapter_batch, metadata_batch = [], [], [], []

    print(f"NUM DATAPOINTS IN FILE {total_datapoints}")
    return (data_batches, paragraph_batches, chapter_batches), metadata_batches, positive_datapoints, total_datapoints


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