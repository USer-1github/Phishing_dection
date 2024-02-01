# import ch_ml as chml
from sklearn.ensemble import RandomForestClassifier 
import Feature_extraction_ff1 as fex
import model

# best_model = RandomForestClassifier(n_estimators=60)
# best_model.fit(chml.X, chml.y)
def detect_phishing(domain):
    new_url_features = fex.data_set_list_creation(domain)


    new_url_features = [new_url_features]  


    prediction = model.pred(new_url_features)

    print("last logicr predictiuon --> ",prediction)
    if prediction >= 1:
        print("The URL  is precdicted as a phishing URL.")
    else:
        print("The URL  is predicted as a legitimate URL.")
