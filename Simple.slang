


void copy(float3 src, out float4 dest)
{
    dest = float4(src, 1.0);
}

// Add an amount to a given pixel
void brighten(float3 amount, inout float4 pixel)
{
    pixel.rgb += amount;
}