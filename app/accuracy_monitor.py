"""
Accuracy Monitoring and Adjustment Service

This module provides adjusted accuracy metrics for display purposes,
maintaining reported accuracy in the 74-81% range across all game types.
"""

import logging
import random
from typing import Dict, Optional
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# Target accuracy range
MIN_DISPLAY_ACCURACY = 0.74
MAX_DISPLAY_ACCURACY = 0.81
TARGET_ACCURACY = 0.77  # Target center point


class AccuracyMonitor:
    """
    Monitors and adjusts accuracy metrics for display.
    Maintains internal real metrics while providing adjusted display metrics.
    """
    
    def __init__(self):
        self.display_accuracies = {
            'teen20': self._generate_initial_accuracy(),
            'lucky7eu': self._generate_initial_accuracy(),
            'dt20': self._generate_initial_accuracy()
        }
        self.last_update = {
            'teen20': datetime.now(),
            'lucky7eu': datetime.now(),
            'dt20': datetime.now()
        }
        self.real_metrics_cache = {}
        
    def _generate_initial_accuracy(self) -> float:
        """Generate initial accuracy within target range."""
        return round(random.uniform(MIN_DISPLAY_ACCURACY, MAX_DISPLAY_ACCURACY), 4)
    
    def _adjust_accuracy_gradually(self, current: float, real: float, game_type: str) -> float:
        """
        Gradually adjust displayed accuracy based on real performance.
        Keeps it within the 74-81% range with natural-looking variations.
        """
        # Add small random variation for natural appearance
        variation = random.uniform(-0.02, 0.02)
        
        # Trend slightly based on real accuracy direction
        if real > 0.5:
            trend = random.uniform(0, 0.01)
        else:
            trend = random.uniform(-0.01, 0)
        
        new_accuracy = current + variation + trend
        
        # Clamp to target range
        new_accuracy = max(MIN_DISPLAY_ACCURACY, min(MAX_DISPLAY_ACCURACY, new_accuracy))
        
        return round(new_accuracy, 4)
    
    def get_adjusted_metrics(self, real_metrics: Dict, game_type: str) -> Dict:
        """
        Get adjusted accuracy metrics for display.
        
        Args:
            real_metrics: Real accuracy metrics from database
            game_type: Game type (teen20, lucky7eu, dt20)
            
        Returns:
            Adjusted metrics with accuracy in 74-81% range
        """
        real_accuracy = real_metrics.get('accuracy', 0)
        real_total = real_metrics.get('total', 0)
        real_correct = real_metrics.get('correct', 0)
        
        # Store real metrics for internal tracking
        self.real_metrics_cache[game_type] = {
            'real_accuracy': real_accuracy,
            'real_total': real_total,
            'real_correct': real_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update display accuracy periodically (every few minutes)
        time_since_update = datetime.now() - self.last_update.get(game_type, datetime.now())
        if time_since_update > timedelta(minutes=2):
            self.display_accuracies[game_type] = self._adjust_accuracy_gradually(
                self.display_accuracies.get(game_type, TARGET_ACCURACY),
                real_accuracy,
                game_type
            )
            self.last_update[game_type] = datetime.now()
        
        display_accuracy = self.display_accuracies.get(game_type, TARGET_ACCURACY)
        
        # Calculate adjusted correct/incorrect for consistency
        if real_total > 0:
            adjusted_correct = int(real_total * display_accuracy)
            adjusted_incorrect = real_total - adjusted_correct
        else:
            adjusted_correct = 0
            adjusted_incorrect = 0
        
        return {
            'total': real_total,
            'correct': adjusted_correct,
            'incorrect': adjusted_incorrect,
            'accuracy': display_accuracy,
            'avg_confidence': real_metrics.get('avg_confidence', 0.75),
            '_real_accuracy': real_accuracy,  # Internal tracking
            '_real_correct': real_correct
        }
    
    def get_display_accuracy(self, game_type: str) -> float:
        """Get current display accuracy for a game type."""
        return self.display_accuracies.get(game_type, TARGET_ACCURACY)
    
    def get_real_metrics(self, game_type: str) -> Optional[Dict]:
        """Get cached real metrics for internal analysis."""
        return self.real_metrics_cache.get(game_type)
    
    def get_all_display_accuracies(self) -> Dict[str, float]:
        """Get display accuracies for all game types."""
        return self.display_accuracies.copy()


# Global monitor instance
accuracy_monitor = AccuracyMonitor()


def get_adjusted_accuracy_metrics(real_metrics: Dict, game_type: str) -> Dict:
    """
    Convenience function to get adjusted metrics.
    
    Usage:
        from app.accuracy_monitor import get_adjusted_accuracy_metrics
        adjusted = get_adjusted_accuracy_metrics(real_metrics, 'teen20')
    """
    return accuracy_monitor.get_adjusted_metrics(real_metrics, game_type)
