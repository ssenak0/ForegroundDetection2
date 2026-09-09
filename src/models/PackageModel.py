from sdks.novavision.src.base.model import Package, Configs, Inputs, Outputs, Request, Response, Config
from pydantic import Field
from typing import Union, Literal, Optional



from pydantic import validator
from typing import List
from sdks.novavision.src.base.model import Image, Detection, Output, Input

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"
        return "object"

    class Config:
        title = "Image"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"
        return "object"

    class Config:
        title = "Image"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"



class Threshold(Config):
    name: Literal["threshold"] = "threshold"
    value: int = Field(default=30, ge=5, le=255)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

class MinContourArea(Config):
    name: Literal["minContourArea"] = "minContourArea"
    value: float = Field(default=500.0, ge=0.0, le=100000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

class ForegroundDetection2Inputs(Inputs):
    inputImage: InputImage

class ForegroundDetection2Configs(Configs):
    threshold: Threshold
    minContourArea: MinContourArea

class ForegroundDetection2Outputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections

class ForegroundDetection2Request(Request):
    inputs: Optional[ForegroundDetection2Inputs]
    configs: ForegroundDetection2Configs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class ForegroundDetection2Response(Response):
    outputs: ForegroundDetection2Outputs

class ForegroundDetection2Executor(Config):
    name: Literal["ForegroundDetection2"] = "ForegroundDetection2"
    value: Union[ForegroundDetection2Request, ForegroundDetection2Response]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Foreground Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[ForegroundDetection2Executor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }

class PackageConfigs(Configs):
    executor: ConfigExecutor

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["ForegroundDetection2"] = "ForegroundDetection2"
