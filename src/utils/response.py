from capsules.ForegroundDetection2.src.models.PackageModel import PackageModel, ConfigExecutor, PackageConfigs, OutputDetections, ForegroundDetection2Outputs, ForegroundDetection2Response, ForegroundDetection2Executor, OutputImage

def build_response(context):
    outputImage = OutputImage(
        value=context.image.value,
        dataType=context.image.dataType,
        metaData=context.image.metaData
    )
    outputDetections = OutputDetections(
        value=context.detections,
    )
    Outputs = ForegroundDetection2Outputs(outputImage=outputImage, outputDetections=outputDetections)
    res = ForegroundDetection2Response(outputs=Outputs)
    exec_resp = ForegroundDetection2Executor(value=res)
    executor = ConfigExecutor(value=exec_resp)
    
    context.request.model.configs = PackageConfigs(executor=executor)
    return context.request.model.model_dump(by_alias=True)
