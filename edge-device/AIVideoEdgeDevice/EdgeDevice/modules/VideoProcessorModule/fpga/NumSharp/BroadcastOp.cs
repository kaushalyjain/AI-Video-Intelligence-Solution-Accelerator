using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public class BroadcastOp
    {
        readonly Shape selfShape;
        readonly Shape rhsShape;
        readonly int shapeMaxIndex;
        readonly int[] dimensions;
        readonly int[] rhsDimensions;
        readonly int[] strides;
        int[] indices;

        public static void ValidateBroadcastCompatibleShapes(int selfLength, int rhsLength, Shape selfShape, Shape rhsShape)
        {
            int selfExpectedLength = 1;
            int rhsExpectedLength = 1;

            // Ensure compatible shapes
            // Shapes must be the same length
            if (selfShape.Dimensions.Length != rhsShape.Dimensions.Length)
            {
                throw new ApplicationException("Shapes must be the same length");
            }
            // Shapes dimensions must match or be 1 (broadcastable) for the rhsShape
            for (int i = 0; i < selfShape.Dimensions.Length; i++)
            {
                int selfSize = selfShape.Dimensions[i];
                selfExpectedLength *= selfSize;
                int rhsSize = rhsShape.Dimensions[i];
                rhsExpectedLength *= rhsSize;
                // Broadcast currently only implemented for the final dimension
                if (selfSize != rhsSize && (rhsSize != 1 || i != (selfShape.Dimensions.Length - 1)))
                {
                    throw new ApplicationException("Shapes dimensions must match or be 1 (broadcastable) for the rhsShape");
                }
            }
            if (selfLength != selfExpectedLength)
            {
                throw new ApplicationException("selfStore length does not match selfShape");
            }
            if (rhsLength != rhsExpectedLength)
            {
                throw new ApplicationException("rhsStore length does not match rhsShape");
            }
            if (selfShape.Order != "C" || rhsShape.Order != "C")
            {
                throw new ApplicationException("BroadcastOp is only implemented for both shapes using \"C\" ordering");
            }
        }

        private BroadcastOp(Shape selfShape, Shape rhsShape)
        {
            this.selfShape = selfShape;
            this.rhsShape = rhsShape;
            this.shapeMaxIndex = selfShape.Dimensions.Length - 1;
            this.dimensions = selfShape.Dimensions;
            this.rhsDimensions = rhsShape.Dimensions;
            this.indices = new int[selfShape.Dimensions.Length];
            this.strides = selfShape.Strides;
        }

        private int GetRhsIncrement()
        {
            // This trivialized implementation only broadcasts the final dimension
            int i = this.shapeMaxIndex;
            for (; i >= 0; i--)
            {
                this.indices[i] = this.indices[i] + 1;
                if (this.indices[i] == this.dimensions[i])
                {
                    // A carry is necessary, so keep looping
                    this.indices[i] = 0;
                }
                else
                {
                    // No carry needed, so we terminate the loop
                    break;
                }
            }

            // This trivialized implementation only broadcasts the final dimension
            if (i == this.shapeMaxIndex)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        public static void SelfSubtractFloat(NDArray self, NDArray rhs)
        {
            SelfSubtractFloat(self.GetData<float>(), rhs.GetData<float>(), self.shape, rhs.shape);
        }

        public static NDArray SubtractFloat(NDArray lhs, NDArray rhs)
        {
            NDArray result = lhs.copy();
            SelfSubtractFloat(result, rhs);
            return result;
        }

        public static void SelfMultiplyFloat(NDArray self, NDArray rhs)
        {
            float[] selfStore = self.GetData<float>();
            float[] rhsStore = rhs.GetData<float>();
            ValidateBroadcastCompatibleShapes(selfStore.Length, rhsStore.Length, self.shape, rhs.shape);

            int idxR = 0;
            BroadcastOp op = new BroadcastOp(self.shape, rhs.shape);

            for (int i = 0; i < selfStore.Length; i++)
            {
                selfStore[i] = selfStore[i] * rhsStore[idxR];
                idxR += op.GetRhsIncrement();
            }
        }

        public static void SelfDivideFloat(NDArray self, NDArray rhs)
        {
            float[] selfStore = self.GetData<float>();
            float[] rhsStore = rhs.GetData<float>();
            ValidateBroadcastCompatibleShapes(selfStore.Length, rhsStore.Length, self.shape, rhs.shape);

            int idxR = 0;
            BroadcastOp op = new BroadcastOp(self.shape, rhs.shape);

            for (int i = 0; i < selfStore.Length; i++)
            {
                selfStore[i] = selfStore[i] / rhsStore[idxR];
                idxR += op.GetRhsIncrement();
            }
        }

        public static void SelfPowerFloat(NDArray self, double baseValue = Math.E)
        {
            SelfPowerFloat(self.GetData<float>());
        }

        public static void SelfPowerFloat(float[] selfStore, double baseValue = Math.E)
        {
            for (int i = 0; i < selfStore.Length; i++)
            {
                selfStore[i] = (float)Math.Pow(baseValue, selfStore[i]);
            }
        }

        public static void SelfSubtractFloat(float[] selfStore, float[] rhsStore, Shape selfShape, Shape rhsShape)
        {
            ValidateBroadcastCompatibleShapes(selfStore.Length, rhsStore.Length, selfShape, rhsShape);

            int idxR = 0;
            BroadcastOp op = new BroadcastOp(selfShape, rhsShape);

            for (int i = 0; i < selfStore.Length; i++)
            {
                selfStore[i] = selfStore[i] - rhsStore[idxR];
                idxR += op.GetRhsIncrement();
            }
        }

        public static NDArray SumFloat(NDArray self, int axis)
        {
            float[] selfStore = self.GetData<float>();
            // Only implemented for the final axis so far
            if (axis != self.shape.Length - 1)
            {
                throw new ApplicationException("Sum is currently only implemented for the final axis");
            }
            int[] resultDims = new int[self.shape.Length - 1];
            for (int i = 0; i < resultDims.Length; i++)
            {
                resultDims[i] = self.shape[i];
            }

            Shape resultShape = new Shape(resultDims);
            float[] resultValues = new float[resultShape.Size];

            int lastStride = self.shape[self.shape.Length - 1];
            for (int i = 0; i < selfStore.Length; i++)
            {
                resultValues[i / lastStride] += selfStore[i];
            }

            return new NDArray(resultValues, resultShape);
        }
    }
}
