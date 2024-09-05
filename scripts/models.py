
import argparse
import torch
import torch.nn as nn
import torch.optim as optim



class SequenceTagger(nn.Module):

    def __init__(self, input_size, label_classes, label_class_weights = None, hidden_dim = 512, output_layers = 1, lstm_layers = 2, dropout = 0.4):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        # input_size must be the same size as the bert embedding
        assert input_size == 768
        super(SequenceTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim, num_layers = lstm_layers, batch_first = True, bidirectional = True)
        
        self.heads = nn.ModuleList()
        self.classes = label_classes
        self.class_weights = []
        self.dropout = nn.Dropout(p=dropout)

        for i, labels in enumerate(label_classes):
            head = nn.Sequential()
            if labels:
                self.class_weights.append(label_class_weights[i] if label_class_weights else 1)
                for layer_num in range(output_layers - 1):
                    head.add_module(f"Linear Layer {layer_num}", nn.Linear(in_features = hidden_dim * 2, out_features = hidden_dim * 2))
                    head.add_module(f"Relu {layer_num}", nn.ReLU())
                head.add_module(f"Output layer", nn.Linear(in_features = hidden_dim * 2, out_features = len(labels)))
            else:
                self.class_weights.append(0)
            self.heads.append(head)

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()
        

    def forward(self, sentence_embeds, labels = None, flatten = False, device = "cpu"):
        self.lstm.flatten_parameters() # input is (32, SEQ_LEN, 768)
        lstm_out, _ = self.lstm(sentence_embeds) # lstm_out is (batch_size, seq_len, 2 * hidden_dim)
        lstm_out = self.dropout(lstm_out)
        preds = [output_layer(lstm_out) for output_layer in self.heads]
        
        if labels:
            return (self.get_loss(preds, labels), preds)
        return preds # (N, L, num_classes)
    
    def get_loss(self, preds, labels):
        loss = 0
        for pred, label, label_class, class_weight in zip(preds, labels, self.classes, self.class_weights):
            if label_class and class_weight:
                reshaped_output, reshaped_label = flatten_output_and_label(pred, label, len(label_class))
                loss += class_weight * self.loss_fn(reshaped_output, reshaped_label)
        return loss
    
def flatten_output_and_label(output, label, num_classes=2):
    output = output.view(-1, num_classes) # (N, L, num_classes) -> (N x L, num_classes)
    label = label.view(-1) # (N, L) -> (N x L)
    if num_classes == 2:
        label = (label > 0).long()
    
    return output, label

# if flatten:
# 				return (self.get_loss(preds,))
def flatten_multiclass_outputs_and_labels(task_outputs, task_labels, task_classes):
    assert len(task_classes) == len(task_outputs)
    
    flattened_preds, flattened_labels = [], []

    # Looping over different classification tasks
    for class_lst, output, label in zip(task_classes, task_outputs, task_labels):
        num_classes = len(class_lst)
        
        if num_classes > 0:
            flattened_output, transformed_label = flatten_output_and_label(output, label, num_classes)
            flattened_preds.append(flattened_output)
            flattened_labels.append(transformed_label.cpu().tolist())
        else:
            flattened_preds.append(torch.empty(0, dtype=torch.long))
            flattened_labels.append([])

    return flattened_preds, flattened_labels

class SequenceTaggerWithBahdanauAttention(nn.Module):

    def __init__(self, input_size, num_classes, hidden_dim = 512):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        super(SequenceTaggerWithBahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        attention_input_size = input_size * 2
        self.forward_lstm = nn.LSTM(input_size = attention_input_size, hidden_size = hidden_dim, num_layers = 1, batch_first = True, bidirectional = False)
        self.backward_lstm = nn.LSTM(input_size = attention_input_size, hidden_size = hidden_dim, num_layers = 1, batch_first = True, bidirectional = False)
        self.output = nn.Linear(in_features = hidden_dim * 2, out_features = num_classes)

        # Implement hidden size
        self.encoder_attention = nn.Linear(in_features = input_size, out_features = hidden_dim)
        self.decoder_attention = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, sentence_embeds, device = "cpu"):
        self.forward_lstm.flatten_parameters() # input is (32, SEQ_LEN, 768)
        self.backward_lstm.flatten_parameters()
        batch_size = sentence_embeds.size(0)

        # Forward pass
        forward_outputs = []
        (h_n, c_n) = self.get_initial_states(batch_size, device)

        for t in range(sentence_embeds.size(1)):
            attention_scores = self.calculate_attention_scores(sentence_embeds, h_n)
            attention_weights = torch.softmax(attention_scores, dim = 1) # softmax across seq_len; [batch_size, seq_len]
            context_vector = torch.sum(attention_weights.unsqueeze(dim = 2) * sentence_embeds, dim = 1) # match to [batch_size, seq_len, 768]; weighted sum across embeds in seq
            contextualized_input = torch.cat((context_vector, sentence_embeds[:, t, :]), dim = 1).unsqueeze(dim = 1) # unsqueeze to [N, 1, input_size]
            # print(f"contexted input: {contextualized_input.shape}")
            # print(f"h_n: {h_n.shape}")
            # print(f"c_n: {c_n.shape}")
            output, (h_n, c_n) = self.forward_lstm(contextualized_input, (h_n, c_n))
            forward_outputs.append(output.squeeze(dim = 1)) # convert to [batch, hidden_dim]

        # Backward pass
        backward_outputs = []
        (h_n, c_n) = self.get_initial_states(batch_size, device)

        for t in range(sentence_embeds.size(1)-1, -1, -1):
            attention_scores = self.calculate_attention_scores(sentence_embeds, h_n)
            attention_weights = torch.softmax(attention_scores, dim = 1) # softmax across seq_len; [batch_size, seq_len]
            context_vector = torch.sum(attention_weights.unsqueeze(dim = 2) * sentence_embeds, dim = 1)
            contextualized_input = torch.cat((context_vector, sentence_embeds[:, t, :]), dim = 1).unsqueeze(dim = 1)
            output, (h_n, c_n) = self.backward_lstm(contextualized_input, (h_n, c_n))
            backward_outputs.append(output.squeeze(dim = 1))
        backward_outputs.reverse()

        # Combine forward and backward
        # forward output shape: [batch, hidden_dim]
        combined_outputs = [torch.cat((f, b), dim = 1) for f, b in zip(forward_outputs, backward_outputs)]
        preds = [self.output(result) for result in combined_outputs]
        
        return (torch.stack(preds, dim = 1)) # [batch_size, seq_len, output_classes]
    
    def calculate_attention_scores(self, encoder_embeddings, decoder_hidden):
        # Input sizes:
            # encoder_embeddings: [batch_size, seq_len, input_size]
            # decoder_hidden: [batch_size, decoder_dim]
        enc_proj = self.encoder_attention(encoder_embeddings) # [batch_size, seq_len, hidden_dim]
        dec_proj = self.decoder_attention(decoder_hidden).squeeze(dim = 0) # [batch_size, hidden_dim]

        dec_proj = dec_proj.unsqueeze(dim = 1).expand_as(enc_proj) # expand to [batch_size, seq_len, hidden_dim]

        tanh_result = torch.tanh(enc_proj + dec_proj) # [batch_size, seq_len, hidden_dim]
        pre_scores = self.v * tanh_result # broadcast [hidden_size] to multiply with [batch_size, seq_len, hidden_dim]
        scores = torch.sum(pre_scores, dim = 2) # [batch_size, seq_len]
        return scores
    
    def get_initial_states(self, batch_size, device = "cpu"):
        h_0 = torch.zeros((1, batch_size, self.hidden_dim), device = device)
        c_0 = torch.zeros((1, batch_size, self.hidden_dim), device = device)
        return (h_0, c_0)
    
class GeneralMulticlassSequenceTaggerWithBahdanauAttention(nn.Module):

    def __init__(self, input_size, label_classes, label_class_weights = None, hidden_dim = 512, output_layers = 1, lstm_layers = 1):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        super(GeneralMulticlassSequenceTaggerWithBahdanauAttention, self).__init__()

        # Define bidirectional lstms
        self.hidden_dim = hidden_dim
        attention_input_size = input_size * 2
        self.forward_lstm = nn.LSTM(input_size = attention_input_size, hidden_size = hidden_dim, num_layers = lstm_layers, batch_first = True, bidirectional = False)
        self.backward_lstm = nn.LSTM(input_size = attention_input_size, hidden_size = hidden_dim, num_layers = lstm_layers, batch_first = True, bidirectional = False)
        
        # Implement attention layers
        self.encoder_attention = nn.Linear(in_features = input_size, out_features = hidden_dim)
        self.decoder_attention = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

        # Define heads based on class labels and set weights (default 1)
        self.heads = nn.ModuleList()
        self.classes = label_classes
        self.class_weights = []

        for i, labels in enumerate(label_classes):
            head = nn.Sequential()
            if labels:
                self.class_weights.append(label_class_weights[i] if label_class_weights else 1)
                for layer_num in range(output_layers - 1):
                    head.add_module(f"Linear Layer {layer_num}", nn.Linear(in_features = hidden_dim * 2, out_features = hidden_dim * 2))
                    head.add_module(f"Relu {layer_num}", nn.ReLU())
                head.add_module(f"Output layer", nn.Linear(in_features = hidden_dim * 2, out_features = len(labels)))
            else:
                self.class_weights.append(0)
            self.heads.append(head)

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, sentence_embeds, labels = None, device = "cpu"):
        self.forward_lstm.flatten_parameters() # input is (32, SEQ_LEN, 768)
        self.backward_lstm.flatten_parameters()
        batch_size = sentence_embeds.size(0)
        print(f"Batch size in forward: {batch_size}")
        
        # Begin forward LSTM pass
        forward_outputs = []

        # Set initial state
        (h_n, c_n) = self.get_initial_states(batch_size, device)

        for t in range(sentence_embeds.size(1)):
            attention_scores = self.calculate_attention_scores(sentence_embeds, h_n)
            attention_weights = torch.softmax(attention_scores, dim = 1) # softmax across seq_len; [batch_size, seq_len]
            context_vector = torch.sum(attention_weights.unsqueeze(dim = 2) * sentence_embeds, dim = 1) # match to [batch_size, seq_len, 768]; weighted sum across embeds in seq
            contextualized_input = torch.cat((context_vector, sentence_embeds[:, t, :]), dim = 1).unsqueeze(dim = 1) # unsqueeze to [N, 1, input_size]
            output, (h_n, c_n) = self.forward_lstm(contextualized_input, (h_n, c_n))
            forward_outputs.append(output.squeeze(dim = 1)) # convert to [batch, hidden_dim]

        # Begin backward LSTM pass
        backward_outputs = []

        # Set initial state
        (h_n, c_n) = self.get_initial_states(batch_size, device)

        for t in range(sentence_embeds.size(1)-1, -1, -1):
            attention_scores = self.calculate_attention_scores(sentence_embeds, h_n)
            attention_weights = torch.softmax(attention_scores, dim = 1) # softmax across seq_len; [batch_size, seq_len]
            context_vector = torch.sum(attention_weights.unsqueeze(dim = 2) * sentence_embeds, dim = 1)
            contextualized_input = torch.cat((context_vector, sentence_embeds[:, t, :]), dim = 1).unsqueeze(dim = 1)
            output, (h_n, c_n) = self.backward_lstm(contextualized_input, (h_n, c_n))
            backward_outputs.append(output.squeeze(dim = 1))
        backward_outputs.reverse()

        # Combine forward and backward --> forward output shape: [batch, hidden_dim]
        combined_outputs = [torch.cat((f, b), dim = 1) for f, b in zip(forward_outputs, backward_outputs)]
        preds = [torch.stack([output_layer(result) for result in combined_outputs], dim = 1) for output_layer in self.heads] # [batch_size, seq_len, output_classes]
        
        if labels:
            return (self.get_loss(preds, labels), preds)
                
        return preds
    
    def get_loss(self, preds, labels):
        loss = 0
        for pred, label, label_class, class_weight in zip(preds, labels, self.classes, self.class_weights):
            if label_class and class_weight:
                reshaped_output, reshaped_label = flatten_output_and_label(pred, label, len(label_class))
                loss += class_weight * self.loss_fn(reshaped_output, reshaped_label)
        return loss
    
    def calculate_attention_scores(self, encoder_embeddings, decoder_hidden):
        # Input sizes:
            # encoder_embeddings: [batch_size, seq_len, input_size]
            # decoder_hidden: [batch_size, decoder_dim]
        enc_proj = self.encoder_attention(encoder_embeddings) # [batch_size, seq_len, hidden_dim]
        dec_proj = self.decoder_attention(decoder_hidden).squeeze(dim = 0) # [batch_size, hidden_dim]

        dec_proj = dec_proj.unsqueeze(dim = 1).expand_as(enc_proj) # expand to [batch_size, seq_len, hidden_dim]

        tanh_result = torch.tanh(enc_proj + dec_proj) # [batch_size, seq_len, hidden_dim]
        pre_scores = self.v * tanh_result # broadcast [hidden_size] to multiply with [batch_size, seq_len, hidden_dim]
        scores = torch.sum(pre_scores, dim = 2) # [batch_size, seq_len]
        return scores
    
    def get_initial_states(self, batch_size, device = "cpu"):
        h_0 = torch.zeros((1, batch_size, self.hidden_dim), device = device)
        c_0 = torch.zeros((1, batch_size, self.hidden_dim), device = device)
        return (h_0, c_0)
    

def flatten_output_and_label(output, label, num_classes=2):
    output = output.view(-1, num_classes) # (N, L, num_classes) -> (N x L, num_classes)
    label = label.view(-1) # (N, L) -> (N x L)
    if num_classes == 2:
        label = (label > 0).long()
    
    return output, label


class MulticlassSequenceTaggerWithBahdanauAttention(nn.Module):

    def __init__(self, input_size, label_classes, hidden_dim = 512, output_layers = 1, lstm_layers = 1):
        # input: (N, L, H_in), output: (N, L, D * H_out) where D = 2 if bidirectional, 1 otherwise
        super(MulticlassSequenceTaggerWithBahdanauAttention, self).__init__()

        # Define bidirectional lstms
        self.hidden_dim = hidden_dim
        attention_input_size = input_size * 2
        self.forward_lstm = nn.LSTM(input_size = attention_input_size, hidden_size = hidden_dim, num_layers = lstm_layers, batch_first = True, bidirectional = False)
        self.backward_lstm = nn.LSTM(input_size = attention_input_size, hidden_size = hidden_dim, num_layers = lstm_layers, batch_first = True, bidirectional = False)
        
        # Implement attention layers
        self.encoder_attention = nn.Linear(in_features = input_size, out_features = hidden_dim)
        self.decoder_attention = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

        # Define heads for paragraphs and chapters
        self.paragraph_classes, self.chapter_classes = label_classes
        self.paragraph_head = nn.Sequential()
        self.chapter_head = nn.Sequential()

        for layer_num in range(output_layers - 1):
            self.paragraph_head.add_module(f"Linear Layer {layer_num}", nn.Linear(in_features = hidden_dim * 2, out_features = hidden_dim * 2))
            self.paragraph_head.add_module(f"Relu {layer_num}", nn.Relu())
            self.chapter_head.add_module(f"Linear Layer {layer_num}", nn.Linear(in_features = hidden_dim * 2, out_features = hidden_dim * 2))
            self.chapter_head.add_module(f"Relu {layer_num}", nn.Relu())

        self.paragraph_head.add_module(f"Output layer", nn.Linear(in_features = hidden_dim * 2, out_features = len(self.paragraph_classes)))
        self.chapter_head.add_module(f"Output layer", nn.Linear(in_features = hidden_dim * 2, out_features = len(self.chapter_classes)))

        # Define loss function                         
        self.loss_fn = nn.CrossEntropyLoss()

        
    def forward(self, sentence_embeds, labels = None, device = "cpu"):
        self.forward_lstm.flatten_parameters() # input is (32, SEQ_LEN, 768)
        self.backward_lstm.flatten_parameters()
        batch_size = sentence_embeds.size(0)
        print(f"Batch size in forward: {batch_size}")
        if labels:
            print(f"Labels: {len(labels)}")
        
        # Forward pass
        forward_outputs = []
        (h_n, c_n) = self.get_initial_states(batch_size, device)

        for t in range(sentence_embeds.size(1)):
            attention_scores = self.calculate_attention_scores(sentence_embeds, h_n)
            attention_weights = torch.softmax(attention_scores, dim = 1) # softmax across seq_len; [batch_size, seq_len]
            context_vector = torch.sum(attention_weights.unsqueeze(dim = 2) * sentence_embeds, dim = 1) # match to [batch_size, seq_len, 768]; weighted sum across embeds in seq
            contextualized_input = torch.cat((context_vector, sentence_embeds[:, t, :]), dim = 1).unsqueeze(dim = 1) # unsqueeze to [N, 1, input_size]
            # print(f"contexted input: {contextualized_input.shape}")
            # print(f"h_n: {h_n.shape}")
            # print(f"c_n: {c_n.shape}")
            output, (h_n, c_n) = self.forward_lstm(contextualized_input, (h_n, c_n))
            forward_outputs.append(output.squeeze(dim = 1)) # convert to [batch, hidden_dim]

        # Backward pass
        backward_outputs = []
        (h_n, c_n) = self.get_initial_states(batch_size, device)

        for t in range(sentence_embeds.size(1)-1, -1, -1):
            attention_scores = self.calculate_attention_scores(sentence_embeds, h_n)
            attention_weights = torch.softmax(attention_scores, dim = 1) # softmax across seq_len; [batch_size, seq_len]
            context_vector = torch.sum(attention_weights.unsqueeze(dim = 2) * sentence_embeds, dim = 1)
            contextualized_input = torch.cat((context_vector, sentence_embeds[:, t, :]), dim = 1).unsqueeze(dim = 1)
            output, (h_n, c_n) = self.backward_lstm(contextualized_input, (h_n, c_n))
            backward_outputs.append(output.squeeze(dim = 1))
        backward_outputs.reverse()

        # Combine forward and backward
        # forward output shape: [batch, hidden_dim]
        combined_outputs = [torch.cat((f, b), dim = 1) for f, b in zip(forward_outputs, backward_outputs)]
        preds = [self.output(result) for result in combined_outputs]
        
        return (torch.stack(preds, dim = 1)) # [batch_size, seq_len, output_classes]
    
    def get_loss(self, preds, labels):
        loss = 0
        for pred, label, label_class in zip(preds, labels, self.classes):
            reshaped_output, reshaped_label = flatten_output_and_label(pred, label, len(label_class))
            print(reshaped_label)
            loss += self.loss_fn(reshaped_output, reshaped_label)
        return loss
    
    def calculate_attention_scores(self, encoder_embeddings, decoder_hidden):
        # Input sizes:
            # encoder_embeddings: [batch_size, seq_len, input_size]
            # decoder_hidden: [batch_size, decoder_dim]
        enc_proj = self.encoder_attention(encoder_embeddings) # [batch_size, seq_len, hidden_dim]
        dec_proj = self.decoder_attention(decoder_hidden).squeeze(dim = 0) # [batch_size, hidden_dim]

        dec_proj = dec_proj.unsqueeze(dim = 1).expand_as(enc_proj) # expand to [batch_size, seq_len, hidden_dim]

        tanh_result = torch.tanh(enc_proj + dec_proj) # [batch_size, seq_len, hidden_dim]
        pre_scores = self.v * tanh_result # broadcast [hidden_size] to multiply with [batch_size, seq_len, hidden_dim]
        scores = torch.sum(pre_scores, dim = 2) # [batch_size, seq_len]
        return scores
    
    def get_initial_states(self, batch_size, device = "cpu"):
        h_0 = torch.zeros((1, batch_size, self.hidden_dim), device = device)
        c_0 = torch.zeros((1, batch_size, self.hidden_dim), device = device)
        return (h_0, c_0)