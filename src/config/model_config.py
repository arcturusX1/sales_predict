"""
Model configuration management for training and inference.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""
    
    model_type: str  # 'catboost' or 'xgboost'
    model_name: str  # Unique identifier
    version: str  # Semantic version
    training_period: Optional[Dict] = None  # {'year': 2024, 'month': 11} or None for full data
    test_size: float = 0.2
    
    # CatBoost hyperparameters
    catboost_iterations: int = 1000
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 8
    
    # XGBoost hyperparameters
    xgboost_n_estimators: int = 1000
    xgboost_learning_rate: float = 0.05
    xgboost_max_depth: int = 8
    xgboost_subsample: float = 0.8
    xgboost_colsample_bytree: float = 0.8
    xgboost_reg_alpha: float = 0.5
    xgboost_reg_lambda: float = 0.5
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    def get_hyperparams(self):
        """Get hyperparameters for the configured model type."""
        if self.model_type == 'catboost':
            return {
                'iterations': self.catboost_iterations,
                'learning_rate': self.catboost_learning_rate,
                'depth': self.catboost_depth,
            }
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': self.xgboost_n_estimators,
                'learning_rate': self.xgboost_learning_rate,
                'max_depth': self.xgboost_max_depth,
                'subsample': self.xgboost_subsample,
                'colsample_bytree': self.xgboost_colsample_bytree,
                'reg_alpha': self.xgboost_reg_alpha,
                'reg_lambda': self.xgboost_reg_lambda,
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create config from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_json(self, json_path: str):
        """Save config to JSON file."""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ModelRegistry:
    """Registry for managing multiple model configurations."""
    
    def __init__(self, registry_file='./models/registry.json'):
        """
        Initialize registry.
        
        Args:
            registry_file: Path to registry JSON file
        """
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.configs = {}
        self._load()
    
    def _load(self):
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                self.configs = {k: ModelConfig.from_dict(v) for k, v in data.items()}
    
    def _save(self):
        """Save registry to file."""
        with open(self.registry_file, 'w') as f:
            data = {k: v.to_dict() for k, v in self.configs.items()}
            json.dump(data, f, indent=2)
    
    def register(self, config: ModelConfig):
        """Register a model configuration."""
        key = f"{config.model_type}_{config.model_name}"
        self.configs[key] = config
        self._save()
    
    def get(self, model_type: str, model_name: str):
        """Get a model configuration."""
        key = f"{model_type}_{model_name}"
        return self.configs.get(key)
    
    def list_models(self):
        """List all registered models."""
        return list(self.configs.keys())
    
    def get_by_type(self, model_type: str):
        """Get all models of a specific type."""
        return [cfg for cfg in self.configs.values() if cfg.model_type == model_type]


# Predefined configurations
XGBOOST_FULL_DATA = ModelConfig(
    model_type='xgboost',
    model_name='full_data',
    version='1.0.0',
    training_period=None,  # Uses all data
    test_size=0.2
)

XGBOOST_JAN_2025 = ModelConfig(
    model_type='xgboost',
    model_name='jan_2025',
    version='1.0.0',
    training_period={'year': 2025, 'month': 1},
    test_size=0.2
)

CATBOOST_FULL_DATA = ModelConfig(
    model_type='catboost',
    model_name='full_data',
    version='1.0.0',
    training_period=None,  # Uses all data
    test_size=0.2
)

CATBOOST_JAN_2025 = ModelConfig(
    model_type='catboost',
    model_name='jan_2025',
    version='1.0.0',
    training_period={'year': 2025, 'month': 1},
    test_size=0.2
)
