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

            ExtractDetections(networkOutput.GetRange(0, 6), networkOutput.GetRange(6, 6), g_ssdAnchors, selectThreshold, nmsThreshold);



            return result;
        }

        /// <summary>
        /// Perform an element-wise operation on source and store the result in dest. 
        /// The source and operand shapes must match except for the last dimension;
        /// this gives a "scalar-ish" o
        /// </summary>
        /// <typeparam name="TD">The Storage array type for the destination</typeparam>
        /// <typeparam name="TS">The Storage array type for the source</typeparam>
        /// <typeparam name="TO">The Storage array type for the operand</typeparam>
        /// <param name="dest">The Storage array for the destination</param>
        /// <param name="source">The Storage array for the source</param>
        /// <param name="operand">The Storage array for the operand</param>
        /// <param name="destShape"></param>
        /// <param name="sourceShape"></param>
        /// <param name="operandShape"></param>
        /// <param name="op">The operation to perform</param>
        /// <remarks>This implementation</remarks>
        static void SingleProjectionOperation<TD, TS, TO>(TD[] dest, TS[] source, TO[] operand, 
            Shape destShape, Shape sourceShape, Shape operandShape, Func<TS, TO, TD> op)
        {
            if (destShape != sourceShape)
            {
                throw new Exception("Source and destination shapes are not equal");
            }
            if (destShape.Dimensions.Length != operandShape.Dimensions.Length)
            {
                throw new Exception("Source and operand shape lengths are not equal");
            }
            for (int i = 0; i < destShape.Dimensions.Length; i++)
            {

            }
        }

        static void ExtractDetections(List<NDArray> predictions, List<NDArray> localizations,
            List<(NDArray, NDArray, NDArray, NDArray)> ssdAnchors, float selectThreshold, float nmsThreshold)
        {

            for (int i = 0; i < predictions.Count; i++)
            {
                SoftMax(predictions[i], axis: 4);
            }
        }

        // This version of SoftMax operates on the source
        static void SoftMax(NDArray x, int axis)
        {
            var max = np.max(x, axis: axis);
            // This gratuitous reshape works around a bug in NumSharp 0.10.4 Shape that produces incorrect size.
            max = np.reshape(max, max.shape);
            var zzzz = max[0][0];
            //max.size
            var maxX = np.expand_dims(max, axis);
            //var delta = x - maxX;
            //NDArray e_x = np.power((x - np.expand_dims(np.max(x, axis: axis), axis)), (float)Math.E);
            //e_x = np.exp(x - np.expand_dims(np.max(x, axis = axis), axis))
            //result = e_x / np.expand_dims(np.sum(e_x, axis = axis), axis)
            //return null;
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
