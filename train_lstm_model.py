from typing import Tuple

import torch
from pandas import DataFrame as df
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer, ngrams_iterator

from NNModel import TextClassifierNN


def yield_tokens(data_iterator, tokenizer, ngrams=1):
    for _, text in data_iterator:
        yield ngrams_iterator(tokenizer(text), ngrams)


def preprocess_data(vocab, tokenizer, X, y) -> Tuple[torch.tensor, torch.tensor]:
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    labels, texts = [], []
    for index in range(len(y)):
        text = X.iloc[index]['text']
        label = y[index]
        texts.append(torch.tensor(text_pipeline(text), dtype=torch.int64))
        labels.append(label_pipeline(label))
    labels = torch.tensor(labels, dtype=torch.int64)
    texts = torch.cat(texts)
    return texts, labels


def train_model(X_train: df, y_train: df, X_val: df, y_val: df, num_classes: int) -> None:
    num_epochs = 5
    embedding_size = 128
    batch_size = 16
    lr = 4.0
    ngram_size = 2
    tokenizer = get_tokenizer('basic_english')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iterator = list(zip(y_train, X_train))
    vocab = build_vocab_from_iterator(yield_tokens(train_iterator, tokenizer, ngram_size), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    texts_train, labels_train = preprocess_data(vocab, tokenizer, X_train, y_train)
    texts_val, labels_val = preprocess_data(vocab, tokenizer, X_val, y_val)

    vocab_size = len(vocab)
    model = TextClassifierNN(vocab_size, embedding_size, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=-0.1)

    train_dataloader = DataLoader(list(zip(labels_train, texts_train)), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(list(zip(labels_val, texts_val)), batch_size=batch_size, shuffle=True)

    total_accuracy = None
    for epoch in range(num_epochs):
        print("Current epoch: {}".format(epoch + 1))
        model.train()
        total_acc, total_count = 0, 0
        for index, (label, text) in enumerate(train_dataloader):
            optimizer.zero_grad()
            pred = model(text)
            loss = criterion(pred, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (pred.argmax(1) == label).sum().item()
            total_count += label.size(0)
            # print("Training loss: {}".format(loss))
        print("Training accuracy: {}".format(total_acc / total_count))

        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for index, (label, text) in enumerate(val_dataloader):
                pred = model(text)
                loss = criterion(pred, label)
                total_acc += (pred.argmax(1) == label).sum().item()
                total_count += label.size(0)
        validation_accuracy = total_acc / total_count
        print("Validation accuracy: {}".format(validation_accuracy))

        if total_accuracy is not None and total_acc > validation_accuracy:
            scheduler.step()
        else:
            total_accuracy = validation_accuracy