import matplotlib.pyplot as plt

import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig, AdamW, get_scheduler
from typing import Tuple, Union
import httpx
from bs4 import BeautifulSoup
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import pdb

class ArtemisDataset(Dataset):
    def __init__(self,
                 processor,
                 labels_fname="/home/shagund/598finalproj/Contrastive.csv",
                 img_dirname="/home/shagund/598finalproj/imgs"):

        dataset_size = 5000

        self.img_dirname = img_dirname
        self.processor = processor
        invalid_images = []
        with open('/home/shagund/598finalproj/imgs/invalid_images.txt', 'r') as infile:
            for line in infile:
                invalid_images.append(int(line.strip()))
        img_labels = pd.read_csv(labels_fname, nrows=dataset_size)

        valid_image_filenames = os.listdir(img_dirname)[:dataset_size]
        img_labels = img_labels[(img_labels['painting']+'.jpeg').isin(valid_image_filenames)]

        self.img_labels = img_labels[~img_labels.index.isin(invalid_images)]
        self.processor = processor
        self.vocab = list(self.processor.tokenizer.get_vocab().keys())

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        painting_name = self.img_labels.iloc[idx]['painting'] + '.jpeg'
        img_path = os.path.join(self.img_dirname, painting_name)
        # need to make sure that we are reading an image that's valid
        image = read_image(img_path)
        label = self.img_labels.iloc[idx]['utterance']
        emotion = self.img_labels.iloc[idx]['emotion']
        # label= f"This is a caption that invokes {emotion}: {label}"
        label = label
        ind = int(len(label)/4)
        prompt = f"Q:Can you continue this caption invoking {emotion} for this image? A:" + label[:ind]
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = label
        encoding["input_prompt"] = prompt
        encoding["emotion"] = emotion
        return encoding

def collate_fn(batch):
  # pad the input_ids and attention_mask
  processed_batch = {}
  for key in batch[0].keys():
      if key != "text" and key != "input_prompt" and key != "emotion":
          processed_batch[key] = torch.stack([example[key] for example in batch])
      elif key == "text":
          text_inputs = processor.tokenizer(
              [example["text"] for example in batch], padding=True, truncation=True, max_length = 55, return_tensors="pt"
          )
          processed_batch["labels"] = text_inputs["input_ids"]

      elif key == "input_prompt":
          prompt_inputs = processor.tokenizer(
              [example["input_prompt"] for example in batch], padding=True, truncation = True, max_length = 55, return_tensors="pt"
          )
          processed_batch["input_ids"] = prompt_inputs["input_ids"]
          processed_batch["attention_mask"] = prompt_inputs["attention_mask"]
      elif key == "emotion":
          prompt_emotions = [example["emotion"] for example in batch]
          processed_batch["emotions"] = prompt_emotions
  return processed_batch

# # download the image files themselves (the original ZIP FILE IS 25G so we'll scrape a small subset lol)
import httpx
from bs4 import BeautifulSoup

base_url = "https://uploads8.wikiart.org/images/"
dataset_size = 10000
valid_images_count = 0
invalid_images=[]

df = pd.read_csv('Contrastive.csv')

for i, item in enumerate(df.iterrows()):
    item_name_spl = item[1]['painting'].split('_')
    painter = item_name_spl[0]
    painting = item_name_spl[1]
    scrape_url = base_url + painter + '/' + painting + '.jpg!Large.jpg'

    try:
        response = httpx.get(scrape_url)
        if response.status_code == 200 and response.headers['Content-Type'].startswith('image'):
            with open('imgs/' + item[1]['painting'] + '.jpeg', 'wb') as outf:
                outf.write(response.content)
            valid_images_count += 1
            print(f"Downloaded image {valid_images_count}: {item[1]['painting']}")
        else:
            print(f"Skipped invalid image URL: {scrape_url}")
            invalid_images.append(i)
    except httpx.RequestError as e:
        print(f"Error downloading image: {scrape_url}")
        print(f"Error message: {str(e)}")

    if valid_images_count == dataset_size:
        break

with open('imgs/invalid_images.txt', 'w') as outfile:
    for number in invalid_images:
        outfile.write(str(number) + '\n')

# import shutil

# shutil.make_archive(base_name='download_imgs', format='zip', base_dir='imgs')

class Blip2PromptTuner:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        n_tokens: int=None,
        initialize_from_vocab: bool=True,
        random_range: float=0.5,
    ):
        # from transformers import BitsAndBytesConfig
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16
        # )

        model = super().from_pretrained(pretrained_model, device_map="auto") #device_map={"":0},) #quantization_config=bnb_config)
        for param in model.parameters():
            param.requires_grad = False
        print("initizalizing soft prompts...")
        model.init_soft_prompt_embeds(n_tokens = 8, initialize_from_vocab = True)
        return model

    def init_soft_prompt_embeds(self,
                                n_tokens: int = 8,
                                initialize_from_vocab: bool = True,
                                random_range: float = 0.5,
                                ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = torch.nn.functional.normalize(
                self.language_model.get_input_embeddings().weight[:n_tokens].clone().detach(),
                dim=0
            )
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, 768)#config.n_embd)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids, emotions) -> torch.Tensor:
        # def count_nan_weights():
        # total_nan_count = 0
        # for name, param in self.model.named_parameters():
        #     nan_count = torch.isnan(param.data).sum().item()
        #     if nan_count > 0:
        #         print(f"{name} has {nan_count} NaN values")
        #     total_nan_count += nan_count
        # return total_nan_count
        self.emotion_to_index = {
        'amusement': 0,
        'awe': 1,
        'contentment': 2,
        'excitement': 3,
        'anger': 4,
        'disgust': 5,
        'fear': 6,
        'sadness': 7
        }
        emotion_indices = torch.tensor([self.emotion_to_index[label] for label in emotions], dtype=torch.long, device=input_ids.device)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids).to(self.device) #self.transformer.wte(input_ids) #model.language_model.get_input_embeddings().weight
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        combined_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return combined_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask, extend):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, extend), 1).to(self.device), attention_mask.to(self.device)],
            dim=1,
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        labels = None,
        return_dict = None,
        emotions = None,
    ):

        from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput

        # start of the hugging face forward function

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values.to(self.device),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
    #   inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
    #   inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        input_ids = input_ids.to(self.device)

        if input_ids is not None: # this represents the textual input embeddings with the concat. soft prompts
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids, emotions).to(
            self.device
        )
        extend = inputs_embeds.shape[1] - attention_mask.shape[1]
        inputs_embeds = torch.cat([inputs_embeds.to(language_model_inputs.device), language_model_inputs], dim=1)
        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask, extend).to("cuda")
        attention_mask = torch.cat([attention_mask, language_model_attention_mask], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                # Flatten the tokensS
                loss_fct = nn.CrossEntropyLoss(size_average = True, ignore_index = -100, reduction="mean")
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        # add to device
        loss = loss.to(self.device)
        logits = logits.to(self.device)

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

class BLIPPromptTunerFinal(Blip2PromptTuner, Blip2ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

class Config:
    # Same default parameters as run_clm_no_trainer.py in tranformers
    # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py
    num_train_epochs = 50
    weight_decay = 0.01
    learning_rate = 0.001
    lr_scheduler_type = "linear"
    num_warmup_steps = 0
    max_train_steps = num_train_epochs

    # Prompt-tuning
    # number of prompt tokens
    n_prompt_tokens = 8
    # If True, soft prompt will be initialized from vocab
    # Otherwise, you can set `random_range` to initialize by randomization.
    init_from_vocab = True
    # random_range = 0.5
args = Config()
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = BLIPPromptTunerFinal.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", args.n_prompt_tokens, initialize_from_vocab=True)

def count_nan_weights():
    total_nan_count = 0
    nan_count = torch.isnan(model.soft_prompt.weight).sum().item()
    print(f"soft prompts have {nan_count} NaN values")
    total_nan_count += nan_count
    return total_nan_count

losses = []

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
       # "weight_decay": args.weight_decay,
    }
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-3)
full_dataset = ArtemisDataset(processor=processor)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

model.train()
for epoch in range(50):
  print("epoch: ", epoch)
  for idx, batch in enumerate(train_dataloader):
    batch_loss = model(**batch)
    loss = batch_loss.loss
    print('loss: ',loss)
    losses.append(loss.item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), max_norm=1.0)
    optimizer.step()
    count_nan_weights()
    for name, param in model.soft_prompt.named_parameters():
        if torch.isnan(param.data).any():
            nan_mask = torch.isnan(model.soft_prompt.weight)
            nan_vals = model.soft_prompt.weight[nan_mask]
    optimizer.zero_grad()

    if idx % 4 == 0:
      torch.cuda.empty_cache()

import matplotlib.pyplot as plt

# x-axis
epoch_lbls = [int(i) for i in list(range(50))]

plt.plot(epoch_lbls, losses, marker='o', linestyle='-', )
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Training Epochs -- 5k Dataset - Partial Cap. Input')

# Show plot
plt.savefig('/home/shagund/598finalproj/lossplot.png')

"""Evaluation"""


import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from nltk import ngrams, pos_tag

from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def calculate_diversity_metrics(texts):
    # Flatten the list of texts into a single list of tokens
    all_tokens = [token for text in texts for token in text.split()]
    unigrams = list(ngrams(all_tokens, 1))
    bigrams = list(ngrams(all_tokens, 2))

    # Calculate Distinct-1 and Distinct-2
    distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
    return distinct_1, distinct_2

def calculate_bleu(references, hypotheses):
    """
    Calculate BLEU score between actual and predicted sentences using smoothing.
    """
    smoothie = SmoothingFunction().method1  # Using method1 for example
    references_tokenized = [[word_tokenize(ref)] for ref in references]
    hypotheses_tokenized = [word_tokenize(hyp) for hyp in hypotheses]
    return corpus_bleu(references_tokenized, hypotheses_tokenized, smoothing_function=smoothie)

def calculate_meteor(references, hypotheses):
    """
    Calculate METEOR score between actual and predicted sentences.
    """
    references_tokenized = [word_tokenize(ref) for ref in references]
    hypotheses_tokenized = [word_tokenize(hyp) for hyp in hypotheses]
    return sum(meteor_score([ref], hyp) for ref, hyp in zip(references_tokenized, hypotheses_tokenized)) / len(hypotheses_tokenized)

def calculate_rouge(references, hypotheses):
    """
    Calculate ROUGE scores between actual and predicted sentences.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {key: 0.0 for key in ['rouge1', 'rouge2', 'rougeL']}
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        for key in scores:
            scores[key] += score[key].fmeasure
    for key in scores:
        scores[key] /= len(hypotheses)
    return scores

def calculate_cider(references, hypotheses):
    """
    Calculate CIDEr score between actual and predicted sentences.
    """
    scorer = Cider()
    # Convert to the format expected by CIDEr score
    hypo_dict = {i: [hyp] for i, hyp in enumerate(hypotheses)}
    ref_dict = {i: [ref] for i, ref in enumerate(references)}
    score, _ = scorer.compute_score(ref_dict, hypo_dict)
    return score

# Commented out IPython magic to ensure Python compatibility.
# %cd -

model.eval()

# for comparison
golds = []
gens = []

with torch.no_grad():
  for idx, batch in enumerate(test_dataloader):

    gold = processor.tokenizer.batch_decode(batch['labels']) # gold

    output = model(**batch, return_dict=True)
    token_ids = torch.argmax(output.logits, dim=-1)
    gen = processor.tokenizer.batch_decode(token_ids, skip_special_tokens=True) # gen

    golds.extend(gold)
    gens.extend(gen)

# apply evaluation
print('bleu: ',calculate_bleu(golds, gens))
print('meteor: ',calculate_meteor(golds, gens))
print('rouge: ',calculate_rouge(golds, gens))
print('cider: ',calculate_cider(golds, gens))