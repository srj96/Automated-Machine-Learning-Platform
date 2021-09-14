from cerberus import Validator
import pandas as pd 
import numpy as np 

def val_schema(data):
    schema = {}
    for col in data.columns :
        print(col)
        schema_1 = {col : {'type' : 'list', 'items' : [{'type' : not('integer')}]}}
        schema.update(schema_1)
    print(schema)
    return(schema) 

def val_check(doc,schema_form):

    v = Validator(schema_form)

    if v.validate(doc):
        print("True")
    else:
        print("False")

if __name__ == "__main__":

    df = pd.read_csv("/home/congnitensor/Python_projects/data_validation/BostonHousing.csv")

    doc = {}
    for col in df.columns:
        df_series = pd.Series(df[col])
        d = {col : list(df_series.values)}
        doc.update(d)
    
    print(doc)

    schema = val_schema(df)

    # val_check(doc,schema)

    for key,value in doc.items():
        document = {key:value}
        print(document) 
        val_check(document,schema) 
        
        
     





