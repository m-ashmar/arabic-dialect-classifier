import logging
from data_loader import DataLoader
from preprocessor import TextPreprocessor
from model import DialectClassifier
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
    
    try:
        # Load data
        logger.info("بدء تحميل البيانات... | Starting data loading...")
        loader = DataLoader()
        datasets = loader.load_full_dataset()

        # Text preprocessing
        logger.info("بدء معالجة النصوص... | Starting text preprocessing...")
        preprocessor = TextPreprocessor()
        for split in ['train', 'validation', 'test']:
            datasets[split]['clean_text'] = datasets[split]['text'].apply(
                lambda x: preprocessor.full_preprocess(x)
            ).fillna("")

        # Initialize and train classifier
        logger.info("بدء تدريب النموذج... | Starting model training...")
        classifier = DialectClassifier()
        classifier.train(
            datasets['train']['clean_text'],
            datasets['train']['city']
        )

        # Evaluate model
        logger.info("بدء تقييم النموذج... | Starting model evaluation...")
        results = classifier.evaluate(
            datasets['validation']['clean_text'],
            datasets['validation']['city']
        )
        logger.info(f"نتائج التقييم | Evaluation Results: {results}")


    except Exception as e:
        logger.error(f"فشل في الخطة العملية: {str(e)} | Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()