'''
https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb
'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
from datasets import Dataset, ClassLabel, Value


def compute_metrics(eval_pred):
    metric = evaluate.load('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

    
def train_transformer(train_df, eval_df):
    model_ckp = "distilbert-base-uncased"
    model_name = model_ckp.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_ckp, use_fast=True)


    train_df['label'] = train_df['label'].apply(lambda x: {'yes':1, 'no':0}[x]).astype(int)
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    label_map = ClassLabel(num_classes=2, names=['yes', 'no'])
    train_dataset.features['label'] = label_map
    

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    train_dataset = train_dataset.map(preprocess_function)
    eval_dataset = eval_dataset.map(preprocess_function)

    model = AutoModelForSequenceClassification.from_pretrained(model_ckp, num_labels=2)

    batch_size=1
    training_args = TrainingArguments(
        f"{model_name}-finetuned-amita-flairs",
        output_dir="test_trainer",
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        save_strategy='epoch',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )


    trainer.train()

    model.save()