"""
Tetris Genetic AI Implementation
Based on https://github.com/uiucanh/tetris
"""
import os
import sys
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import pickle

# Add the tetris repository to the path
tetris_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tetris')
if tetris_path not in sys.path:
    sys.path.insert(0, tetris_path)

try:
    from core.utils import get_board_info, action_map, feature_names
    from core.gen_algo import Network, Population, get_score
    TETRIS_AVAILABLE = True
except ImportError as e:
    TETRIS_AVAILABLE = False
    print(f"Tetris genetic AI not available: {e}")

from .ai_api_base import AIAPIConnector

logger = logging.getLogger(__name__)

class TetrisGeneticAI(AIAPIConnector):
    """Tetris Genetic Algorithm AI Provider"""

    def __init__(self, api_key: str = "genetic", model_path: str = None):
        super().__init__(api_key)
        self.model_path = model_path or os.path.join(tetris_path, "models", "best.pkl")
        self.network = None
        self.population = None
        self.is_training = False
        self.generation = 0
        self.best_fitness = 0

        if TETRIS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Tetris genetic AI not available - tetris repository not found")

    def _load_model(self):
        """Load the trained model or create a new one"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading existing model from {self.model_path}")
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)

                if isinstance(saved_data, dict) and 'model' in saved_data:
                    # Load trained network
                    self.network = saved_data['model']
                    self.best_fitness = saved_data.get('fitness', 0)
                    self.generation = saved_data.get('generation', 0)
                else:
                    # Legacy format - direct network
                    self.network = saved_data

                logger.info(f"Model loaded successfully. Generation: {self.generation}, Best Fitness: {self.best_fitness}")
            else:
                logger.info("No existing model found, creating new random network")
                self.network = Network()
                self._save_model()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.network = Network()

    def _save_model(self):
        """Save the current model"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            saved_data = {
                'model': self.network,
                'fitness': self.best_fitness,
                'generation': self.generation,
                'timestamp': time.time()
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(saved_data, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def get_game_features(self, emulator) -> np.ndarray:
        """Extract Tetris game features for neural network input"""
        try:
            # Get screen data
            screen = emulator.get_screen()
            if screen is None:
                return np.zeros(9)

            # Convert to grayscale for Tetris board analysis
            if len(screen.shape) == 3:
                screen_gray = np.mean(screen, axis=2)
            else:
                screen_gray = screen

            # Extract Tetris board (simplified approach)
            # In a full implementation, you'd use memory addresses like the original
            board_height = 20
            board_width = 10
            board = np.zeros((board_height, board_width))

            # Simple board extraction from screen (this is simplified)
            # The original implementation uses specific memory addresses
            for y in range(board_height):
                for x in range(board_width):
                    # Map screen coordinates to board
                    screen_x = 20 + x * 8  # Approximate Tetris board position
                    screen_y = 32 + y * 8

                    if (screen_y < screen_gray.shape[0] and
                        screen_x < screen_gray.shape[1]):
                        # Check if this position has a block (non-black)
                        if screen_gray[screen_y, screen_x] > 50:
                            board[y, x] = 1

            # Extract features similar to the original implementation
            features = self._extract_board_features(board)
            return features

        except Exception as e:
            logger.error(f"Error extracting game features: {e}")
            return np.zeros(9)

    def _extract_board_features(self, board: np.ndarray) -> np.ndarray:
        """Extract board features similar to the original tetris implementation"""
        try:
            # Get column heights
            peaks = []
            for x in range(board.shape[1]):
                column = board[:, x]
                height = 0
                for y in range(len(column) - 1, -1, -1):
                    if column[y] > 0:
                        height = len(column) - y
                        break
                peaks.append(height)

            peaks = np.array(peaks)

            # Calculate features
            features = []

            # 1. Aggregate height
            agg_height = np.sum(peaks)
            features.append(agg_height)

            # 2. Number of holes
            n_holes = 0
            for x in range(board.shape[1]):
                column = board[:, x]
                found_block = False
                for y in range(len(column)):
                    if column[y] > 0:
                        found_block = True
                    elif found_block:
                        n_holes += 1
            features.append(n_holes)

            # 3. Bumpiness
            bumpiness = 0
            for i in range(len(peaks) - 1):
                bumpiness += abs(peaks[i] - peaks[i + 1])
            features.append(bumpiness)

            # 4. Cleared lines (simplified - would need game state)
            features.append(0)

            # 5. Number of pits
            n_pits = 0
            for i in range(len(peaks)):
                if (i == 0 or peaks[i] < peaks[i-1]) and (i == len(peaks)-1 or peaks[i] < peaks[i+1]):
                    n_pits += 1
            features.append(n_pits)

            # 6. Maximum wells
            max_wells = 0
            for i in range(len(peaks)):
                well_depth = 0
                if i > 0 and i < len(peaks) - 1:
                    well_depth = min(peaks[i-1], peaks[i+1]) - peaks[i]
                max_wells = max(max_wells, well_depth)
            features.append(max_wells)

            # 7. Columns with holes
            cols_with_holes = 0
            for x in range(board.shape[1]):
                column = board[:, x]
                found_block = False
                has_hole = False
                for y in range(len(column)):
                    if column[y] > 0:
                        found_block = True
                    elif found_block:
                        has_hole = True
                        break
                if has_hole:
                    cols_with_holes += 1
            features.append(cols_with_holes)

            # 8. Row transitions
            row_transitions = 0
            for y in range(board.shape[0]):
                for x in range(board.shape[1] - 1):
                    if board[y, x] != board[y, x + 1]:
                        row_transitions += 1
            features.append(row_transitions)

            # 9. Column transitions
            col_transitions = 0
            for x in range(board.shape[1]):
                for y in range(board.shape[0] - 1):
                    if board[y, x] != board[y + 1, x]:
                        col_transitions += 1
            features.append(col_transitions)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error extracting board features: {e}")
            return np.zeros(9, dtype=np.float32)

    def get_action(self, screen: np.ndarray, game_state: Dict[str, Any],
                   valid_actions: List[str] = None) -> str:
        """Get action using genetic algorithm"""
        if not TETRIS_AVAILABLE or self.network is None:
            return "RIGHT"  # Default fallback

        try:
            # Extract features
            features = self.get_game_features(game_state.get('emulator'))

            # Get network prediction
            with torch.no_grad():
                action_value = self.network.activate(features)

            # Convert action value to Tetris move
            return self._value_to_action(action_value.item(), valid_actions)

        except Exception as e:
            logger.error(f"Error getting genetic AI action: {e}")
            return "RIGHT"  # Fallback

    def _value_to_action(self, value: float, valid_actions: List[str] = None) -> str:
        """Convert neural network output to game action"""
        # Simple strategy: use the value to decide action type
        if valid_actions is None:
            valid_actions = ['LEFT', 'RIGHT', 'DOWN', 'A', 'B']

        # Use the network value as a heuristic for action selection
        # Positive values = more aggressive, Negative values = more defensive
        if value > 0.5:
            # Aggressive - move quickly and drop
            return "DOWN" if "DOWN" in valid_actions else "RIGHT"
        elif value > 0:
            # Moderate - move right
            return "RIGHT" if "RIGHT" in valid_actions else "LEFT"
        elif value > -0.5:
            # Defensive - move left
            return "LEFT" if "LEFT" in valid_actions else "RIGHT"
        else:
            # Very defensive - rotate
            return "A" if "A" in valid_actions else "RIGHT"

    def train_generation(self, emulator, population_size: int = 20,
                        generations: int = 10) -> Dict[str, Any]:
        """Train a new generation of genetic models"""
        if not TETRIS_AVAILABLE:
            return {"error": "Tetris genetic AI not available"}

        try:
            self.is_training = True
            results = {
                "generation": self.generation,
                "population_size": population_size,
                "generations_trained": 0,
                "best_fitness": self.best_fitness,
                "models_evaluated": 0
            }

            logger.info(f"Starting genetic training. Population: {population_size}, Generations: {generations}")

            for gen in range(generations):
                self.generation += 1

                # Create population
                if self.population is None:
                    self.population = Population(size=population_size)
                else:
                    self.population = Population(size=population_size, old_population=self.population)

                # Evaluate population
                fitness_scores = []
                for i, model in enumerate(self.population.models):
                    fitness = self._evaluate_model(model, emulator)
                    fitness_scores.append(fitness)
                    results["models_evaluated"] += 1

                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.network = model

                self.population.fitnesses = np.array(fitness_scores)
                results["generations_trained"] += 1

                logger.info(f"Generation {self.generation}: Best fitness = {max(fitness_scores):.2f}, "
                           f"Avg fitness = {np.mean(fitness_scores):.2f}")

                # Save best model
                if self.best_fitness > results["best_fitness"]:
                    self._save_model()
                    results["best_fitness"] = self.best_fitness

            self.is_training = False
            return results

        except Exception as e:
            logger.error(f"Error during genetic training: {e}")
            self.is_training = False
            return {"error": str(e)}

    def _evaluate_model(self, model, emulator, max_steps: int = 100) -> float:
        """Evaluate a single model's fitness"""
        try:
            # Reset emulator to clean state
            emulator.reset()

            # Simulate game play
            total_score = 0
            steps = 0

            while steps < max_steps:
                # Get screen and features
                features = self.get_game_features(emulator)

                # Get model action
                with torch.no_grad():
                    action_value = model.activate(features)

                action = self._value_to_action(action_value.item())

                # Execute action
                emulator.step(action, 1)

                # Simple fitness calculation
                total_score += self._calculate_fitness(features, action_value.item())
                steps += 1

            return total_score / max_steps

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return 0.0

    def _calculate_fitness(self, features: np.ndarray, action_value: float) -> float:
        """Calculate fitness based on board features and action"""
        try:
            # Reward lower aggregate height
            height_penalty = features[0] * 0.1

            # Penalty for holes
            hole_penalty = features[1] * 2.0

            # Penalty for bumpiness
            bumpiness_penalty = features[2] * 0.1

            # Bonus for cleared lines
            line_bonus = features[3] * 10.0

            # Combine penalties and bonuses
            fitness = line_bonus - height_penalty - hole_penalty - bumpiness_penalty

            # Add action value as a small bonus/penalty
            fitness += action_value * 0.1

            return max(0, fitness)  # Ensure non-negative fitness

        except Exception as e:
            logger.error(f"Error calculating fitness: {e}")
            return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the genetic AI"""
        return {
            "available": TETRIS_AVAILABLE,
            "model_loaded": self.network is not None,
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "is_training": self.is_training,
            "model_path": self.model_path,
            "tetris_available": TETRIS_AVAILABLE
        }

    def save_training_state(self, filepath: str):
        """Save the complete training state"""
        try:
            state = {
                "network": self.network,
                "population": self.population,
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "timestamp": time.time()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Training state saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving training state: {e}")
            return False

    def load_training_state(self, filepath: str):
        """Load a complete training state"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.network = state["network"]
            self.population = state.get("population")
            self.generation = state.get("generation", 0)
            self.best_fitness = state.get("best_fitness", 0)

            logger.info(f"Training state loaded from {filepath}")
            logger.info(f"Generation: {self.generation}, Best Fitness: {self.best_fitness}")
            return True

        except Exception as e:
            logger.error(f"Error loading training state: {e}")
            return False