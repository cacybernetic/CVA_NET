import os
import logging
import pathlib
import shutil
from typing import List, Optional, Dict, Any
# from .model import JEPATrainer, Config
# from .repository import save_config, save_data, load_config, load_data


LOGGER = logging.getLogger(__name__)


class CheckpointManager:
    """
    Gestionnaire de checkpoint qui coordonne la sauvegarde et le chargement
    via des repositories specialises avec gestion automatique du nombre
    de checkpoints conserves.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", max_to_keep: int = 5) -> None:
        """
        Initialise le gestionnaire de checkpoint.

        :param checkpoint_dir: Dossier racine pour les checkpoints
        :param max_to_keep: Nombre maximum de checkpoints a conserver
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep

        from .repository import save_config, save_data, load_config, load_data
        self._trainer_save_config = save_config
        self._trainer_save_data = save_data
        self._trainer_load_config = load_config
        self._trainer_load_data = load_data

    def save_config(self, epoch: int, trainer_config: 'Config') ->  Dict[str, Any]:
        """
        Sauvegarde un checkpoint pour une epoch donnee et nettoie
        les anciens checkpoints si necessaire.

        :param epoch: Numero de l'epoch a sauvegarder.
        :param trainer_config: The trainer config.
        :returns: Chemin vers le dossier de checkpoint cree.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
        pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        # Sauvegarder via les repositories;
        results = self._trainer_save_config(trainer_config, checkpoint_path, encoding='utf-8')
        LOGGER.debug(f"Checkpoint sauvegarde: {checkpoint_path}")
        # Nettoyer les anciens checkpoints
        self._cleanup_old_checkpoints()
        return results

    def save_data(self, epoch: int, trainer: 'JEPATrainer', device_type: str=None) ->  Dict[str, Any]:
        """
        Sauvegarde un checkpoint pour une epoch donnee et nettoie
        les anciens checkpoints si necessaire.

        :param epoch: Numero de l'epoch a sauvegarder.
        :param trainer: The instance of the trainer.
        :param device_type: The device type name.
        :returns: Chemin vers le dossier de checkpoint cree.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
        pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        # Sauvegarder via les repositories;
        results = self._trainer_save_data(trainer, checkpoint_path, device_type=device_type, encoding='utf-8')
        LOGGER.debug(f"Checkpoint sauvegarde: {checkpoint_path}")
        # Nettoyer les anciens checkpoints;
        self._cleanup_old_checkpoints()
        return results

    def load_config(self, epoch: int) -> 'Config':
        """
        Charge un checkpoint pour une epoch donnee.

        :param epoch: Numero de l'epoch a charger.
        :returns: Tuple (modele, optimizer) charges.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trouve: {checkpoint_path}")
        # Charger via les repositories;
        config = self._trainer_load_config(checkpoint_path, encoding='utf-8')
        LOGGER.info(f"Checkpoint charge: {checkpoint_path}")
        return config

    def load_data(self, epoch: int, config: 'Config', trainer: 'JEPATrainer') -> 'JEPATrainer':
        """
        Charge un checkpoint pour une epoch donnee.

        :param epoch: Numero de l'epoch a charger.
        :param config: The training config.
        :param trainer: The instance of the trainer.
        :returns: The same instance of the trainer.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trouve: {checkpoint_path}")
        # Charger via les repositories;
        self._trainer_load_data(checkpoint_path, config, encoding='utf-8', trainer=trainer)
        LOGGER.info(f"Checkpoint charge: {checkpoint_path}")
        return trainer

    def get_latest_checkpoint(self) -> Optional[int]:
        """
        Trouve le numero de l'epoch du dernier checkpoint disponible.

        :returns: Numero de l'epoch du dernier checkpoint, ou None
        """
        checkpoints = self._get_all_checkpoints()
        return max(checkpoints) if checkpoints else None

    def _get_all_checkpoints(self) -> List[int]:
        """
        Retourne la liste de tous les numeros d'epoch des checkpoints existants.

        :returns: Liste des numeros d'epoch des checkpoints
        """
        checkpoints = []
        if not os.path.exists(self.checkpoint_dir):
            return checkpoints
        for item in os.listdir(self.checkpoint_dir):
            if item.startswith("checkpoint_epoch_"):
                try:
                    # Extraire le numero d'epoch du nom du dossier
                    epoch_str = item.split("_")[-1]
                    epoch = int(epoch_str)
                    checkpoints.append(epoch)
                except ValueError:
                    # Ignorer les dossiers qui ne correspondent pas au format
                    continue
        return checkpoints

    def _cleanup_old_checkpoints(self) -> None:
        """
        Supprime les anciens checkpoints pour ne garder que les max_to_keep
        plus recents.
        """
        checkpoints = self._get_all_checkpoints()
        if len(checkpoints) <= self.max_to_keep:
            return  # Rien a nettoyer
        # Trier les checkpoints par ordre croissant (plus anciens en premier)
        checkpoints.sort()
        # Calculer le nombre de checkpoints a supprimer
        num_to_remove = len(checkpoints) - self.max_to_keep
        # Supprimer les checkpoints les plus anciens
        for i in range(num_to_remove):
            epoch_to_remove = checkpoints[i]
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch_to_remove}")
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
                LOGGER.debug(f"Checkpoint ancien supprime: {checkpoint_path}")
