#!/usr/bin/env python3
"""
CNN/DailyMail 数据集预处理脚本
用于 BART 模型适配策略对比研究：全参数微调、超轻量微调、LoRA 与从零训练

预处理步骤：
1. 加载 CNN/DailyMail 数据集
2. 最小化文本清洗（仅标准化空白字符）
3. 使用 BART 分词器进行分词（最大长度 512）
4. 划分训练集、验证集和测试集
5. 保存为 Hugging Face Dataset 格式
"""

from datasets import load_dataset, DatasetDict
from transformers import BartTokenizer, BartModel
import logging
from typing import Dict, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_train_size = 36
_val_size = 6
_test_size = 6

class CNNDailyMailPreprocessor:
    """CNN/DailyMail 数据集预处理类"""
    
    def __init__(self, model_name: str = "facebook/bart-base", max_length: int = 1024):
        """
        初始化预处理器
        
        Args:
            model_name: BART 模型名称
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # 加载 BART 分词器
        logger.info(f"加载分词器：{model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
        logger.info("预处理器初始化完成")
    
    def preprocess_text(self, text: str) -> str:
        """
        最小化文本预处理（仅标准化空白字符）
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        if not text:
            return ""
        
        # 仅标准化空白字符，保留原始大小写、标点和数字
        text = ' '.join(text.split()).strip()
        
        return text
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        分词函数
        
        Args:
            examples: 包含文章和摘要的字典
        
        Returns:
            分词后的字典
        """
        # 最小化预处理文章
        articles = [self.preprocess_text(article) for article in examples['article']]
        
        # 最小化预处理摘要
        highlights = [self.preprocess_text(highlight) for highlight in examples['highlights']]
        
        # 对文章进行分词
        model_inputs = self.tokenizer(
            articles,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length"
        )
        
        # 对摘要进行分词（作为标签）
        labels = self.tokenizer(
            highlights,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def load_and_preprocess(self) -> DatasetDict:
        """
        加载并预处理数据集
        
        Returns:
            预处理后的 DatasetDict
        """
        logger.info(f"加载数据集：cnn_dailymail (版本：3.0.0)")
        
        # 加载数据集
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
        
        # 子集大小（用于快速原型开发）
        train_size = min(_train_size, len(dataset['train']))
        val_size = min(_val_size, len(dataset['validation']))
        test_size = min(_test_size, len(dataset['test']))
        
        # 创建子集
        train_subset = dataset['train'].select(range(train_size))
        val_subset = dataset['validation'].select(range(val_size))
        test_subset = dataset['test'].select(range(test_size))
        
        logger.info(f"训练集大小：{len(train_subset)}")
        logger.info(f"验证集大小：{len(val_subset)}")
        logger.info(f"测试集大小：{len(test_subset)}")
        
        # 应用预处理
        logger.info("开始预处理数据集...")
        
        train_processed = train_subset.map(
            self.tokenize_function,
            batched=True,
            batch_size=8,
            remove_columns=['article', 'highlights', 'id']
        )
        
        val_processed = val_subset.map(
            self.tokenize_function,
            batched=True,
            batch_size=8,
            remove_columns=['article', 'highlights', 'id']
        )
        
        test_processed = test_subset.map(
            self.tokenize_function,
            batched=True,
            batch_size=8,
            remove_columns=['article', 'highlights', 'id']
        )
        
        # 创建 DatasetDict
        processed_datasets = DatasetDict({
            'train': train_processed,
            'validation': val_processed,
            'test': test_processed
        })
        
        logger.info("数据集预处理完成")
        
        return processed_datasets
    
    def save_dataset(self, dataset: DatasetDict, save_path: str = "./processed_cnn_dailymail"):
        """
        保存预处理后的数据集
        
        Args:
            dataset: 预处理后的数据集
            save_path: 保存路径
        """
        logger.info(f"保存数据集到：{save_path}")
        dataset.save_to_disk(save_path)
        logger.info("数据集保存完成")
    
    def get_dataset_info(self, dataset: DatasetDict):
        """显示数据集信息"""
        logger.info("数据集信息:")
        for split in dataset.keys():
            logger.info(f"  {split}: {len(dataset[split])} 个样本")
            
            if len(dataset[split]) > 0:
                sample = dataset[split][0]
                logger.info(f"    样本键：{list(sample.keys())}")
                
                # 解码第一个样本查看
                input_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                label_text = self.tokenizer.decode(sample['labels'], skip_special_tokens=True)
                
                logger.info(f"    输入文本 (前 100 字符): {input_text[:100]}...")
                logger.info(f"    标签文本 (前 100 字符): {label_text[:100]}...")

def process_data():
    """主函数"""
    logger.info("开始 CNN/DailyMail 数据集预处理")
    
    # 初始化预处理器
    preprocessor = CNNDailyMailPreprocessor(
        model_name="facebook/bart-base",
        max_length=512
    )
    
    # 加载并预处理数据集
    processed_datasets = preprocessor.load_and_preprocess()
    
    # 显示数据集信息
    preprocessor.get_dataset_info(processed_datasets)
    
    # 保存数据集
    preprocessor.save_dataset(processed_datasets, "./processed_cnn_dailymail")
    
    logger.info("预处理流程完成")

process_data()