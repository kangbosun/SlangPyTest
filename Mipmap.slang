
import slangpy;
import brdf;

struct MaterialParameters
{
    Tensor<float3, 2> albedo;
    Tensor<float3, 2> normal;
    Tensor<float, 2> roughness;
    float metallic;

    RWTensor<float3, 2> albedo_grad;
    RWTensor<float3, 2> normal_grad;
    RWTensor<float, 2> roughness_grad; 

    [Differentiable]
    float3 get_albedo(int2 pixel)
    {
        return albedo.getv(pixel);
    }

    [BackwardDerivativeOf(get_albedo)]
    void get_albedo_bwd(int2 pixel, float3 grad)
    {
        albedo_grad.setv(pixel, grad);
    }

    [Differentiable]
    float3 get_normal(int2 pixel)
    {
        return normal.getv(pixel);
    }

    [BackwardDerivativeOf(get_normal)]
    void get_normal_bwd(int2 pixel, float3 grad)
    {
        normal_grad.setv(pixel, grad);
    }

    [Differentiable]
    float get_roughness(int2 pixel)
    {
        return roughness.getv(pixel);
    }

    [BackwardDerivativeOf(get_roughness)]
    void get_roughness_bwd(int2 pixel, float grad)
    {
        roughness_grad.setv(pixel, grad);
    }
}

[Differentiable]
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
        material.metallic,
        1.0
    );

    return brdf_eval * light_intensity;
}

// LCG based random number generator that can give us
// uniformly distributed random numbers on a hemisphere.
#define PI 3.14159265358979323846f
struct LCG
{
    uint state;

    __init(uint seed) { state = seed; }

    [mutating]
    uint next_uint()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_float()
    {
        // Convert to float in range [0, 1)
        return (next_uint() >> 8) * 0x1p-24;
    }

    [mutating] float3 next_dir_on_hemisphere()
    {
        // Random azimuthal angle [0, 2π)
        float phi = 2.0f * PI * next_float();

        // Random polar angle [0, π/2) for hemisphere
        float cos_theta = next_float();
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        // Convert to Cartesian coordinates
        return float3(sin_theta * cos(phi),sin_theta * sin(phi),cos_theta);
    }
};

[Differentiable]
float3 loss(int2 pixel, no_diff float3 reference, no_diff MaterialParameters material, no_diff float3 light_dir, no_diff float3 view_dir, no_diff float2 view_scale)
{
    float3 real = render(pixel, material, light_dir, view_dir, view_scale);
    float3 error = reference - real;
    return error * error;
}


void calculate_grad(uint seed, int2 pixel, float3 reference, MaterialParameters material, float3 light_dir, float3 view_dir, float2 view_scale)
{
    //LCG lcg = LCG(seed);

    // float3 light_dir = lcg.next_dir_on_hemisphere();
    // float3 view_dir = lcg.next_dir_on_hemisphere();

    // int2 hi_res_pixel = pixel * 4;

    // float3 sum = 0;

    // for(int x = 0; x < 4; x++)
    // {
    //     for(int y = 0; y < 4; y++)
    //     {
    //         int2 hi_res_pixel_offset = hi_res_pixel + int2(x, y);
    //         sum += render(hi_res_pixel_offset, ref_material, light_dir, view_dir, float2(1.0, 1.0));
    //     }
    // }

    // sum /= 16.0f;

    bwd_diff(loss)(pixel, reference, material, light_dir, view_dir, view_scale, 1);
}

void optimizer_step3(inout float3 mipmap, float3 grad, float learning_rate)
{
    mipmap -= learning_rate * grad;
}

void optimizer_step1(inout float mipmap, float grad, float learning_rate)
{
    mipmap -= learning_rate * grad;
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

void init_normal(inout float3 val)
{
    val = float3(0, 0, 1);
}

void init3(inout float3 val, float3 value)
{
    val = value;
}

void init1(inout float val, float value)
{
    val = value;
}


void clear(float4 color, inout float4 pixel)
{
    pixel = color;
}

