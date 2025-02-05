#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تم إنشاء هذا الملف بواسطة [اسمك] في [التاريخ]
Created by [Your Name] on [Date]

مهمة الملف: تحميل ودمج مجموعات بيانات MADAR وQADI
File Purpose: Loading and merging MADAR/QADI datasets
"""

from pathlib import Path
import pandas as pd
import logging
from typing import Tuple, Dict
import os

# إعداد مسارات الملفات الأساسية
# -------------------------------
# Set base paths for data directories
BASE_DIR = Path(__file__).parent
MADAR_PATH = BASE_DIR   # مسار مجلد بيانات MADAR
QADI_PATH = BASE_DIR     # مسار مجلد بيانات QADI

# إعداد نظام التسجيل (Logging)
# -----------------------------
# Configure logging system
logger = logging.getLogger(__name__)

class DataLoader:
    """محمّل البيانات الرئيسي - يقوم بتحميل ودمج مجموعات البيانات"""
    """Main data loader - handles dataset loading and merging"""
    
    def __init__(self):
        """تهيئة المحمّل مع تعيين المناطق الجغرافية"""
        """Initialize loader with geographic region mapping"""
        self.region_mapping = self._load_region_mapping()
        
    @staticmethod
    def _load_region_mapping() -> pd.DataFrame:
        """
        تحميل تعيين المناطق الجغرافية من ملف إكسل
        Load geographic region mapping from Excel file
        
        المخرجات:
            DataFrame يحتوي على تعيين المدن إلى المناطق
        Returns:
            DataFrame with city to region mapping
        """
        mapping_file = BASE_DIR / "dialect_name_unification.xlsx"
        try:
            return pd.read_excel(mapping_file)
        except FileNotFoundError as e:
            logger.error(f"ملف التعيين غير موجود: {mapping_file} | File not found: {mapping_file}")
            raise

    def _load_madar(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        تحميل بيانات MADAR من ملفات TSV
        Load MADAR data from TSV files
        
        المخرجات:
            ثلاث DataFrames: بيانات التدريب، التحقق، الاختبار
        Returns:
            Three DataFrames: train, validation, test
        """
        madar_dfs = []
        
        # البحث عن جميع ملفات TSV في المجلد المحدد
        # Find all TSV files in specified directory
        for file in MADAR_PATH.glob("*.tsv"):
            try:
                df = pd.read_csv(file, sep="\t")
                df = df.rename(columns={'lang': 'city', 'sent': 'text'})
                madar_dfs.append(df)
                logger.info(f"تم تحميل ملف MADAR: {file.name}")
            except Exception as e:
                logger.error(f"خطأ في تحميل {file.name}: {e}")
        
        if not madar_dfs:
            logger.error("لم يتم تحميل أي بيانات من MADAR | No MADAR data loaded")
            raise ValueError("لا توجد بيانات MADAR للدمج | No MADAR data to concatenate")

        full_df = pd.concat(madar_dfs, ignore_index=True)
        return self._split_dataset(full_df)

    def _load_qadi(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        تحميل بيانات QADI من ملفات Parquet
        Load QADI data from Parquet files
        
        المخرجات:
            ثلاث DataFrames: بيانات التدريب، التحقق، الاختبار
        Returns:
            Three DataFrames: train, validation, test
        """
        splits = {
            'train': "train-00000-of-00001.parquet",
            'validation': "validation-00000-of-00001.parquet",
            'test': "test-00000-of-00001.parquet"
        }
        
        dfs = {}
        for split, file in splits.items():
            try:
                df = pd.read_parquet(QADI_PATH / file, engine="pyarrow")
                df = df.rename(columns={'label': 'country'})
                dfs[split] = df
                logger.info(f"تم تحميل قسم {split} من QADI")
            except Exception as e:
                logger.error(f"خطأ في تحميل {split} من QADI: {e}")
                raise
        
        return dfs['train'], dfs['validation'], dfs['test']

    @staticmethod
    def _split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        تقسيم البيانات إلى مجموعات تدريب/تحقق/اختبار
        Split data into train/validation/test sets
        
        المدخلات:
            df: DataFrame يحتوي على عمود 'split'
        Inputs:
            df: DataFrame with 'split' column
            
        المخرجات:
            ثلاث DataFrames مقسمة حسب النوع
        Returns:
            Three split DataFrames
        """
        return (
            df[df['split'].str.endswith('train')],
            df[df['split'].str.endswith('dev')],
            df[df['split'].str.endswith('test')]
        )

    def load_full_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        دمج مجموعات البيانات مع معالجة القيم المفقودة
        Merge datasets with missing value handling
        
        المخرجات:
            قامة تحتوي على البيانات المدمجة
        Returns:
            Dictionary with merged datasets
        """
        madar_train, madar_val, madar_test = self._load_madar()
        qadi_train, qadi_val, qadi_test = self._load_qadi()
        
        # معالجة القيم المفقودة للمدن
        # Handle missing city values
        for df in [qadi_train, qadi_val, qadi_test]:
            df['city'] = pd.NA
        
        # دمج البيانات من المصدرين
        # Merge data from both sources
        combined_train = pd.concat([madar_train.dropna(), qadi_train.dropna()], ignore_index=True)
        combined_val = pd.concat([madar_val.dropna(), qadi_val.dropna()], ignore_index=True)
        combined_test = pd.concat([madar_test.dropna(), qadi_test.dropna()], ignore_index=True)

       
        # تنظيف الأعمدة النهائية
        # Final column cleaning
        for split in [combined_train, combined_val, combined_test]:
            split['city'] = split['city'].astype('string').fillna('unknown')
            logger.info(f"عدد المدن الفريدة: {split['city'].nunique()}")
        
        return {
            'train': combined_train,
            'validation': combined_val,
            'test': combined_test
        }

# مثال على الاستخدام - تم إنشاؤه بواسطة [اسمك]
# Usage Example - Created by [Your Name]
if __name__ == "__main__":
    data_loader = DataLoader()
    datasets = data_loader.load_full_dataset()
    print(datasets['train'].head())