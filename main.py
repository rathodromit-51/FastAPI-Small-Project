import io
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Media Income of Neighborhood")
    HouseAge: float = Field(gt=0, description="Average House Age")
    AveRooms: float = Field(gt=0, description="Average rooms in neighborhood")
    AveBedrms: float = Field(gt=0, description="Average bed rooms in neighborhood")
    populations : float = Field(gt=0, description="Populations in neighborhood")
    AveOccup: float = Field(gt=0, description="Average persons living in house at neighborhood")
    Latitude: float = Field(ge=32, le=42, description="Latitude ration")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude ratio")
    
    
@app.get('/')
def home():
    return{
        "message":"California house prediction API",
        'status': "Running",
        "Endpoint": "send POST request to /predict "
    }
    
@app.get('/heath')
def health():
    return{
        'status':'running',
        'model': 'RadomForestRegressor',
        'features': features,
        'avg_error':"$ 32,773"
    }
    
    
@app.post('/predict')
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.populations,
            "AveOccup" : house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude,
        }])
        
        predicted= model.predict(input_data)[0]
        price = predicted * 100000
        
        return{
            "predicted_Price": f"{price:,.0f}",
            "predicted_Price_Short": f"{predicted:,.2f} Hundred thousands",
            "finance_range": f"{price - 32773:,.2f} TO {price + 32773:,.2f}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed {str(e)}"
        )
        
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a CSV file"
            )
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_columns = [    
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in CSV file: {', '.join(missing_columns)}"
            )
        
        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty"
            )
        
        try:
            predictions = model.predict(df[required_columns])
            df["predicted_price_usd"] = predictions * 100000            

            df["predicted_price_usd"] = df["predicted_price_usd"].apply(lambda x : f"$ {x:,.0f}")

            output = df.to_csv(index=False)

            return StreamingResponse(
                io.StringIO(output),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=predictions.csv"
                }
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed {str(e)}"
            )
