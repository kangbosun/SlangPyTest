
// A simple function that adds two numbers together
float add(float a, float b)
{
    return a + b;
}

// Add an amount to a given pixel
void brighten(float4 amount, inout float4 pixel)
{
    pixel += amount;
}