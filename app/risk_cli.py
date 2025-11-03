# app/risk_cli.py
import argparse
import numpy as np
from joblib import load

MODEL_PATH = r"C:\Users\marco\PycharmProjects\IMPRINT_pilot\data\imprint_risk_model.joblib"

def classify_risk(p: float) -> str:
    if p < 0.004:
        return "Basso"
    elif p < 0.023:
        return "Intermedio"
    else:
        return "Alto"

def suggest_action(cls: str) -> str:
    if cls == "Alto":
        return "Imaging avanzato/MDT; considerare biopsia/chirurgia con approccio oncologico."
    elif cls == "Intermedio":
        return "Rivalutazione ecografica, ripetere markers; considerare MRI."
    return "Gestione conservativa come leiomioma, follow-up."

def main():
    parser = argparse.ArgumentParser(description="Calcolo rischio malignità IMPRINT")
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--diameter_mm", type=float, required=True)
    parser.add_argument("--irregular_margins", type=int, choices=[0,1], required=True)
    parser.add_argument("--color4", type=int, choices=[0,1], required=True)
    parser.add_argument("--shadows_absent", type=int, choices=[0,1], required=True)
    parser.add_argument("--SII", type=float, required=True)
    parser.add_argument("--SIRI", type=float, required=True)
    parser.add_argument("--NLR", type=float, required=True)
    parser.add_argument("--PIV", type=float, required=True)
    args = parser.parse_args()

    diameter_gt8 = 1 if args.diameter_mm > 80 else 0
    x = np.array([[args.NLR, args.SII, args.SIRI, args.PIV,
                   args.age, diameter_gt8, args.irregular_margins,
                   args.color4, args.shadows_absent]])

    model = load(MODEL_PATH)
    p = float(model.predict_proba(x)[0,1])
    cls = classify_risk(p)
    action = suggest_action(cls)

    print(f"Probabilità di malignità: {p:.4f}")
    print(f"Classe di rischio: {cls}")
    print(f"Suggerimento: {action}")

if __name__ == "__main__":
    main()