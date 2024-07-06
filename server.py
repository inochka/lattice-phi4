from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import json
from pydantic import Base

class ComputationResult(Base)

app = FastAPI()

@app.post("/computation_result/")
async def add_result():

    return JSONResponse(jsonable_encoder(graph.make_list_of_models_for_nodes()))


#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8003)