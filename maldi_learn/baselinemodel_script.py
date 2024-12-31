import json
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle
import os

def load_json_to_dict(json_str):
    try:
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError:
        print("Invalid JSON string")
        return {}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a JSON string as an argument")
    else:
        json_str = sys.argv[1]
        input_dict = load_json_to_dict(json_str)
    
    # check if the model output pickle exists
    model_name = input_dict.get("model") + "_"
    for key in input_dict:
        if key != "model":
            model_name += str(input_dict[key]) + "_"
    
    model_name = model_name[:-1]
    job_name   = model_name
    model_name += ".pkl"
    model_name = "./baseline_models/" + model_name

    sys.stdout=open(f"baseline_models/{job_name}.txt","w")

    if not os.path.isfile(model_name):


        X1 = np.loadtxt("./results_random_end_to_end_one_fit/val_x.csv", delimiter = ",")
        X2 = np.loadtxt("./results_random_end_to_end_one_fit/train_x.csv", delimiter = ",")
        Y1 = np.loadtxt("./results_random_end_to_end_one_fit/val_y.csv", delimiter=",")
        Y2 = np.loadtxt("./results_random_end_to_end_one_fit/train_y.csv", delimiter=",")

        # Concatenate 
        X = np.concatenate((X1, X2), axis=0)
        Y = np.concatenate((Y1, Y2), axis=0)

        # Debugging
        if False:
            X = X[:100]
            Y = Y[:100]

        print(X.shape)
        print(Y.shape)

        if input_dict.get("model") == "SVM":
            model = OneVsRestClassifier(
                SVC(C=input_dict.get("C"), 
                    kernel=input_dict.get("kernel"), 
                    gamma=input_dict.get("gamma"),
                    probability=True,
                    verbose=True),
                    n_jobs=2
                    )

        if input_dict.get("model") == "GBM":
            model = OneVsRestClassifier(
                HistGradientBoostingClassifier(
                    max_iter=input_dict.get("max_iter"), 
                    max_depth=input_dict.get("max_depth"),
                    verbose=True),
                    n_jobs=4
            )    
        
        if input_dict.get("model") == "RF":
            model = RandomForestClassifier(n_estimators=input_dict.get("n_estimators"), 
                                        max_depth=input_dict.get("max_depth"),
                                        n_jobs=1,
                                        verbose=True)

        # Train the model
        model.fit(X, Y)

        with open(model_name, "wb") as f:
            pickle.dump(model, f)

    else:
        with open(model_name, "rb") as f:
            model = pickle.load(f)

    Y_test = np.loadtxt("./results_random_end_to_end_one_fit/test_y.csv", delimiter=",")
    X_test = np.loadtxt("./results_random_end_to_end_one_fit/test_x.csv", delimiter=",")
    
    Y_pred = model.predict(X_test)
    Y_pred_proba = model.predict_proba(X_test)

    print(Y_pred_proba)

    # Save the predictions according to model name
    Y_pred_name = model_name[:-4] + "_pred.csv"
    Y_pred_proba_name = model_name[:-4] + "_pred_proba.csv"

    np.savetxt(Y_pred_name, Y_pred, delimiter=",")
    np.savetxt(Y_pred_proba_name, Y_pred_proba, delimiter=",")

    print("Model saved as: ", model_name)
    print("Predictions saved as: ", Y_pred_name)
    print("Predictions probabilities saved as: ", Y_pred_proba_name)
    
    sys.stdout.close()


