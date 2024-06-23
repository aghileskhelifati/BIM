import cv2
import numpy as np  # numpy est un package qui optimise les array (tableaux)
from dotenv import load_dotenv
import os
import aiohttp
import asyncio
import torch

# Chargement des variables d'environnement à partir d'un fichier .env
load_dotenv()

# Chargement du modèle YOLOv5 pré-entraîné
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Définition des classes de YOLOv5
CLASSES = model.names

# Définition des objets spécifiques pour l'ordinateur portable et la souris
pc_object = {"laptop"}
mouse_object = {"mouse"}

# Fonction asynchrone pour afficher les flux de la caméra
async def display_camera_streams():

    # Initialisation de la capture vidéo par la webcam du PC
    cap = cv2.VideoCapture(0)

    # Vérification de l'ouverture de la caméra
    if not cap.isOpened():
        print("Erreur d'ouverture de la caméra réseau")
        return
    
    frame_count = 0
    while True:
        # Lire une image de la webcam
        ret, frame = cap.read()

        # Vérification de la réussite de la capture de l'image
        if not ret:
            print("erreur dans la capture vidéo")
            break

        # Utilisation de YOLOv5 pour la détection des objets dans l'image
        results = model(frame)

        # Initialisation d'un dictionnaire pour compter les occurrences des objets détectés
        counts = {}
        # Récupération des résultats de détection
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            label = CLASSES[int(cls)]
            counts[label] = counts.get(label, 0) + 1

            # Définition des couleurs pour les encadrements en fonction du type d'objet
            if label in pc_object:
                color = (0, 255, 0)  # vert pour les ordinateurs portables
            elif label in mouse_object:
                color = (0, 255, 255)  # jaune pour les souris
            else:
                color = (255, 255, 255)  # blanc par défaut
            
            # Dessiner un rectangle autour des objets détectés et ajouter une étiquette
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Afficher le nombre d'occurrences en haut à gauche de la vidéo
        y_offset = 30
        for label, count in counts.items():
            if count > 0:
                cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30

        # Afficher l'image avec les détections
        cv2.imshow('Camera Stream - YOLOv5 Object Detection', frame)

        frame_count += 1

        # Arrêter le flux vidéo si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('k'):
            break

    # Libération de la capture vidéo et fermeture des fenêtres ouvertes
    cap.release()
    cv2.destroyAllWindows()

# Fonction principale asynchrone
async def main():
    await display_camera_streams()

# Exécution de la fonction principale si ce fichier est exécuté en tant que script
if __name__ == "__main__":
    asyncio.run(main())
