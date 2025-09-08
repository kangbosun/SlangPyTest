
import slangpy;
import brdf;

struct MaterialParameters
{
    Tensor<float3, 2> albedo;
    Tensor<float3, 2> normal;
    Tensor<float, 2> roughness;
    float metallic;

    float3 get_albedo(int2 pixel)
    {
        return albedo.getv(pixel);
    }

    float3 get_normal(int2 pixel)
    {
        return normal.getv(pixel);
    }

    float get_roughness(int2 pixel)
    {
        return roughness.getv(pixel);
    }
}

float3 render(int2 pixel, MaterialParameters material, float3 light_dir, float3 view_dir, float2 view_scale)
{
    float light_intensity = 1.5;
    int2 scaled_pixel = int2(pixel.x * view_scale.x, pixel.y * view_scale.y);

    float3 brdf_eval = eval_brdf(
        material.get_albedo(scaled_pixel),
        normalize(light_dir),
        normalize(view_dir),
        normalize(material.get_normal(scaled_pixel)),
        material.get_roughness(scaled_pixel),
        //0.05,
        material.metallic,
        1.0
    );

    return brdf_eval * light_intensity;
}

float3 downsample3(int2 pixel, Tensor<float3, 2> source)
{
    float3 res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}

float downsample1(int2 pixel, Tensor<float, 2> source)
{
    float res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}



void clear(float4 color, inout float4 pixel)
{
    pixel = color;
}

float3 loss(float3 reference, float3 real)
{
    float3 error = reference - real;
    return error * error;
}