using System;
using System.IO;

public static class IDXReader
{
    // Reads an IDX file and returns a multidimensional array of bytes
    public static Array ReadIDX(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);

        // Read magic number
        int magic = ReadBigEndianInt32(br);
        int dataType = (magic >> 8) & 0xFF;
        int numDims = magic & 0xFF;

        // Read dimensions
        int[] dims = new int[numDims];
        for (int i = 0; i < numDims; i++)
            dims[i] = ReadBigEndianInt32(br);

        // Only support unsigned byte (0x08) for simplicity
        if (dataType != 0x08)
            throw new NotSupportedException("Only unsigned byte IDX files are supported.");

        // Read data
        int total = 1;
        foreach (var d in dims) total *= d;
        byte[] flat = br.ReadBytes(total);

        // Create multidimensional array
        Array arr = Array.CreateInstance(typeof(byte), dims);
        int[] idx = new int[numDims];
        for (int i = 0; i < total; i++)
        {
            int offset = i;
            for (int d = numDims - 1; d >= 0; d--)
            {
                idx[d] = offset % dims[d];
                offset /= dims[d];
            }
            arr.SetValue(flat[i], idx);
        }
        return arr;
    }

    private static int ReadBigEndianInt32(BinaryReader br)
    {
        byte[] bytes = br.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }
}