/* model_ops.inc — MobileNet-1D baseline — operações do modelo (8) */
    resolver->AddConv2D();   /* CONV_2D */
    resolver->AddDepthwiseConv2D();   /* DEPTHWISE_CONV_2D */
    resolver->AddExpandDims();   /* EXPAND_DIMS */
    resolver->AddFullyConnected();   /* FULLY_CONNECTED */
    resolver->AddLogistic();   /* LOGISTIC */
    resolver->AddMean();   /* MEAN */
    resolver->AddPad();   /* PAD */
    resolver->AddReshape();   /* RESHAPE */
