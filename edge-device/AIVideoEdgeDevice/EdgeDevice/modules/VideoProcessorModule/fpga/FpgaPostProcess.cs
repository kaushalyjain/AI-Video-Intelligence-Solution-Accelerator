using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Serving;
using NumSharp;
using VideoProcessorModule;

namespace FpgaClient
{
    public partial class FpgaPostProcess
    {
        static readonly List<(NDArray, NDArray, NDArray, NDArray)> g_ssdAnchors;

        static string[] tensorOutputs =
        {
            "ssd_300_vgg/block4_box/Reshape_1:0",
            "ssd_300_vgg/block7_box/Reshape_1:0",
            "ssd_300_vgg/block8_box/Reshape_1:0",
            "ssd_300_vgg/block9_box/Reshape_1:0",
            "ssd_300_vgg/block10_box/Reshape_1:0",
            "ssd_300_vgg/block11_box/Reshape_1:0",
            "ssd_300_vgg/block4_box/Reshape:0",
            "ssd_300_vgg/block7_box/Reshape:0",
            "ssd_300_vgg/block8_box/Reshape:0",
            "ssd_300_vgg/block9_box/Reshape:0",
            "ssd_300_vgg/block10_box/Reshape:0",
            "ssd_300_vgg/block11_box/Reshape:0"
        };

        static FpgaPostProcess()
        {
            g_ssdAnchors = ComputeAnchors();
        }

        public static List<ImageFeature> PostProcess(PredictResponse networkOutput_raw, float selectThreshold, float nmsThreshold)
        {
            List<NDArray> networkOutput = new List<NDArray>();
            foreach(string str in tensorOutputs)
            {
                NDArray ndarray = NDArrayEx.FromTensorProto(networkOutput_raw.Outputs[str]);
                if (ndarray.shape.Length != 5)
                {
                    ndarray = np.expand_dims(ndarray, axis: 0);
                    if (ndarray.shape.Length != 5)
                    {
                        throw new ApplicationException($"NDArray shape length expected 5, is {ndarray.shape.Length}");
                    }
                }
                networkOutput.Add(ndarray);
            }

            List<ImageFeature> result = new List<ImageFeature>();

            ExtractDetections(networkOutput.GetRange(0, 6), networkOutput.GetRange(6, 6), g_ssdAnchors, 
                selectThreshold, nmsThreshold, imageShape: (300, 300), numClasses: 21);



            return result;
        }

        static void ExtractDetections(List<NDArray> predictions, List<NDArray> localizations,
            List<(NDArray, NDArray, NDArray, NDArray)> ssdAnchors, 
            float selectThreshold, float nmsThreshold, (int, int) imageShape, int numClasses)
        {

            for (int i = 0; i < predictions.Count; i++)
            {
                NDArray preds = SoftMax(predictions[i], axis: 4);
                SelectLayerBoxes(preds, localizations[i], ssdAnchors[i], 
                    selectThreshold, nmsThreshold, imageShape, numClasses);
            }
        }

        static void SelectLayerBoxes(
            NDArray predictions, 
            NDArray localizations,
            (NDArray, NDArray, NDArray, NDArray) anchors,
            float selectThreshold,
            float nmsThreshold,
            (int, int) imageShape,
            int numClasses
            )
        {

        }

        static NDArray SoftMax(NDArray x, int axis)
        {
            // Original Python
            //e_x = np.exp(x - np.expand_dims(np.max(x, axis = axis), axis))
            //result = e_x / np.expand_dims(np.sum(e_x, axis = axis), axis)
            //return result

            var max = np.max(x, axis: axis);
            // This gratuitous reshape works around a bug in NumSharp 0.10.4 Shape that produces incorrect size.
            max = np.reshape(max, max.shape);
            var maxExpanded = np.expand_dims(max, axis);
            var e_xMinusMaxExpanded = BroadcastOp.SubtractFloat(x, maxExpanded);
            BroadcastOp.SelfPowerFloat(e_xMinusMaxExpanded);

            var sum = BroadcastOp.SumFloat(e_xMinusMaxExpanded, axis: axis);
            var sumExpanded = np.expand_dims(sum, axis);
            BroadcastOp.SelfDivideFloat(e_xMinusMaxExpanded, sumExpanded);

            return e_xMinusMaxExpanded;
        }
    }
}
/*
   has_batch_dim = (len(network_output[0].shape) == 5)
    if not has_batch_dim:
        for out in network_output:
            out = np.expand_dims(out, axis=0)
            assert len(out.shape) == 5

*/
