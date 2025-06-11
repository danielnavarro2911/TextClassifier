from huggingface_hub import login
from google.colab import userdata,drive
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification



class TextClassifier:
    def __init__(self,model_path):
        '''
        Clase para predecir texto segun los labels introducidos usando bart-large-mnli.
        Este debe estar guardado en una carpeta de google drive

        Link: https://huggingface.co/facebook/bart-large-mnli

        Ejemplo:

        tc = TextClassifier(model_path)
        
        text = "one day I will see the world"
        labels = ['travel', 'cooking']

        tc.predict(text,labels,umbral = 0.5) -> retorna 'travel'

        una vez los resultados se pueden visualizar los resultados generales con self.general_results

        '''

        API_KEY = userdata.get('HF')
        login(API_KEY)

        drive.mount('/content/drive')


        self.model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.classifier = pipeline("zero-shot-classification",model=model,tokenizer=tokenizer)
    def predict(self,text:str,labels:list, umbral=0.9):
        results = self.classifier(text,labels)
                
        self.general_results = {results['labels'][i]: results['scores'][i] for i in range(len(results['scores']))}

        for label,score in self.general_results.items():
            if score > umbral:
                return label
