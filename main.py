import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import numpy as np
# from torch.utils.tensorboard import SummaryWrite
from tensorboardX import SummaryWriter
import time
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

train_data = [
    "1+1=2",
]

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.inputs = []
        for text in texts:
            encodings = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
            self.inputs.append(encodings)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

import itertools
class ArithmeticDataset(Dataset):
    def __init__(self, operator, max_length):
        self.operator = operator
        self.max_length = max_length
        self.data = self._generate_dataset()

    def _generate_dataset(self):
        # numbers = range(10**(self.max_length - 1), 10**self.max_length)
        numbers = range(0, 10**self.max_length)
        dataset = []
        for a, b in itertools.product(numbers, repeat=2):
            if self.operator == '+':
                c = a + b
            elif self.operator == '-':
                c = a - b
            elif self.operator == '*':
                c = a * b
            else:
                raise ValueError("Invalid operator. Supported: '+', '-', '*'.")
            
            input_str = f"{a}{self.operator}{b}="
            label_str = str(c)
            dataset.append((input_str, label_str))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_str, label_str = self.data[idx]
        return input_str, label_str

class TokenizedArithmeticDataset(Dataset):
    def __init__(self, arithmetic_dataset, tokenizer):
        self.arithmetic_dataset = arithmetic_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.arithmetic_dataset)

    def __getitem__(self, idx):
        input_text, target_text = self.arithmetic_dataset[idx]
        full_text = input_text + target_text + "<|endoftext|>"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        targets = self.tokenizer(full_text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)

        input_ids = targets["input_ids"].squeeze(dim=0)
        labels = input_ids.clone()

        inputs_attention_mask = inputs["attention_mask"].squeeze(dim=0)
        targets_attention_mask = targets["attention_mask"].squeeze(dim=0)
        attention_mask = targets_attention_mask
        for i in range(inputs_attention_mask.size(dim=0)) :
            if inputs_attention_mask[i] == targets_attention_mask[i]:
                labels[i] = -100

        return {
            "input_ids": input_ids,
            # "attention_mask": inputs["attention_mask"].squeeze(),
            "attention_mask": targets_attention_mask,
            "labels": labels
        }



def train(model, train_dataloader, optimizer, scheduler, device, num_epochs, gradient_accumulation_steps, writer: SummaryWriter):
    model.train()
    progress_bar = tqdm(range(num_epochs), desc=f"Train num_epochs={num_epochs}")
    for epoch in progress_bar:
        epoch_loss_accum = 0
        total = 0
        correct = 0

        # progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        # for step, batch in enumerate(progress_bar):
        for step, batch in enumerate(train_dataloader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # labels = inputs.clone()

            # print("attention_mask:", attention_mask[0][0])
            # return
            # labels[:, :, :3] = -100
            # attention_mask[:, :, :4] = 0
            # attention_mask[:, :, :] = 0

            # outputs = model(inputs, labels=inputs)
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_loss_accum += loss.item()
            
            _, predicted = outputs.logits.max(2)
            total += inputs.size(dim=0)
            correct += ((labels[:, 1:] == -100) | (predicted[:, :-1] == labels[:, 1:])).all(dim=1).sum().item()

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # progress_bar.set_postfix({'loss': f"{epoch_loss_accum / (step + 1):.4e}"})
            # progress_bar.set_postfix({'Step': step, 'loss': f"{epoch_loss_accum / (step + 1):.2e}"})
        
        # print((labels == -100))
        # print((predicted == labels))
        # print(labels)
        # print(predicted)

        error = total - correct
        accuracy = (correct / total) if total != 0 else 0
        writer.add_scalar("Loss/Train", epoch_loss_accum / len(train_dataloader), epoch)
        writer.add_scalar("Accuracy/Train", accuracy, epoch)
        writer.add_scalar("Errors/Train", error, epoch)
        writer.add_scalar("LearningRate", scheduler.get_lr(), epoch)
        writer.flush()

        accuracy_percent = int(accuracy*100)
        progress_bar.set_postfix({'Acc': accuracy_percent})

        average_loss = epoch_loss_accum / len(train_dataloader)
        # print(f"Epoch {epoch+1}, Average loss: {average_loss:.4e}, Accuracy: {int(accuracy*100)}%, Error: {error}, Correct: {correct}")

def evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = inputs.clone()

            # outputs = model(inputs, labels=inputs)
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    # input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    encoding = tokenizer(prompt, return_tensors='pt').to(model.device)
    # input_ids = tokenizer.encode(prompt, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt').to(model.device)
    # input_ids = encoding['input_ids'].to(model.device)
    # attention_mask = encoding['attention_mask'].to(model.device)
    # output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)
    # output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, do_sample=True)
    output = model.generate(**encoding, max_length=max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_and_show(model, tokenizer, prompt):
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"Generated text from prompt '{prompt}':")
    print(generated_text)
    return generated_text

def main():

    # Data Prepare
    # train_texts = ["First Sentence.", "Second Sentence.", "Third Sentence."]
    # eval_texts = ["Evaluation First Sentence.", "Evaluation Second Sentence."]
    # train_texts = train_data*(2**7)

    # Load Tokenizer and Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dataset and Dataloader
    # train_dataset = CustomDataset(train_texts, tokenizer, max_length)
    arithmetic_dataset = ArithmeticDataset(operator, length)
    tokenizer_arithmetic_dataset = TokenizedArithmeticDataset(arithmetic_dataset, tokenizer)
    train_dataset = tokenizer_arithmetic_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"step per epoch: {len(train_dataloader)}")

    # eval_dataset = CustomDataset(eval_texts, tokenizer, max_length)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and Scheduler
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=warmup_steps, 
    #     num_training_steps=len(train_dataloader) * num_epochs
    # )
    train_step_size = len(train_dataloader) * num_epochs
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer=optimizer,
    #     eta_min=0,
    #     last_epoch=-1,
    #     T_0=train_step_size + 1,
    #     T_mult=1,
    # )
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=100,
        cycle_mult=2.0,
        max_lr=learning_rate,
        min_lr=0.0,
        warmup_steps=50,
        gamma=1.0)
    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, learning_rate)

    # test_gnerating(model, tokenizer)

    # Train
    print("Starting training...")
    writer = SummaryWriter(log_dir=run_name)
    train(model, train_dataloader, optimizer, scheduler, device, num_epochs, gradient_accumulation_steps, writer)
    writer.close()

    # # Evaluation
    # print("Starting evaluation...")
    # avg_loss, perplexity = evaluate(model, eval_dataloader, device)
    # print(f"Evaluation results: Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    # Text Generation Test
    test_generating(model, tokenizer)

    # Save Model
    print("Saving model...")
    model.save_pretrained("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")
    print("Model saved successfully.")

def test_generating(model, tokenizer):
    print("Generating text...")
    generate_and_show(model, tokenizer, "Hello, ")
    generate_and_show(model, tokenizer, "1")
    generate_and_show(model, tokenizer, "1+")
    generate_and_show(model, tokenizer, "1+1")
    generate_and_show(model, tokenizer, "1+1=")
    generate_and_show(model, tokenizer, "1+1=2")
    generate_and_show(model, tokenizer, "1+1=")
    generate_and_show(model, tokenizer, "1+2=")
    generate_and_show(model, tokenizer, "1+3=")
    generate_and_show(model, tokenizer, "2+1=")
    generate_and_show(model, tokenizer, "3+1=")
    generate_and_show(model, tokenizer, "4+2=")
    generate_and_show(model, tokenizer, "9+9=")


# Hyperparameter
max_length = 128
batch_size = 16
num_epochs = 2**8
# learning_rate = 5e-5
learning_rate = 2**(-11)
warmup_steps = 10 # 1000
gradient_accumulation_steps = 1  # 4

operator = "+"
length = 2

run_name = (
    f"./runs/{time.strftime("%Y-%m-%d_%H-%M-%S")}_"
    f"operator{operator}_"
    f"length{length}_"
    f"batch_size{batch_size}_"
    f"learning_rate{learning_rate:.4e}_cosine_warmup_restart"
)

if __name__ == "__main__":
    main()
