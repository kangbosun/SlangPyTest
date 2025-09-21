
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
float3 render(int2 pixel, MaterialParameters material, float3 light_dir, float3 view_dir)
{
    float light_intensity = 1.5;
    int2 scaled_pixel = int2(pixel.x, pixel.y);

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

[Differentiable]
float3 loss(int2 pixel, no_diff float3 reference, no_diff MaterialParameters material, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float3 real = render(pixel, material, light_dir, view_dir);
    float3 error = reference - real;
    return error * error;
}


float3 calculate_grad(uint seed, int2 pixel, float3 reference, MaterialParameters material, float3 light_dir, float3 view_dir)
{
    float3 current_loss = loss(pixel, reference, material, light_dir, view_dir).x;

    bwd_diff(loss)(pixel, reference, material, light_dir, view_dir, 1);

    return current_loss;
}

void adam_step(inout float3 primal, float3 grad, inout float3 mean, inout float3 variance, float learning_rate, int iteration)
{
    // Default Adam values.
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;
    const float ADAM_WEIGHT_DECAY = 0.001;

    // Running average of gradient and square gradient (== variance)
    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    // The running average is initialized with zeros, and underestimates 
    // the true values at the beginning. Adam divides by a compensation factor
    float3 mHat = mean / (1.0f - pow(ADAM_BETA_1, iteration));
    float3 vHat = variance / (1.0f - pow(ADAM_BETA_2, iteration));
    // Compute the adam step by scaling the mean gradient by the learning rate and the variance estimate
    float3 step = learning_rate * (mHat / (sqrt(vHat) + ADAM_EPSILON) + (ADAM_WEIGHT_DECAY * primal));

    // Take an adam step
    primal -= step;
}

void optimizer_adam3(inout float3 primal, float3 grad, inout float3 mean, inout float3 variance, float learning_rate, int iteration, bool renormalize)
{
    adam_step(primal, grad, mean, variance, learning_rate, iteration);

    if (renormalize)
        primal = normalize(primal);
    else    
        primal = saturate(primal);

}

void optimizer_adam1(inout float primal, float grad, inout float mean, inout float variance, float learning_rate, int iteration)
{
    float3 primal3 = float3(primal, 0, 0);
    float3 grad3 = float3(grad, 0, 0);
    float3 mean3 = float3(mean, 0, 0);
    float3 variance3 = float3(variance, 0, 0);

    adam_step(primal3, grad3, mean3, variance3, learning_rate, 1);

    primal = saturate(primal3.x);
    mean = mean3.x;
    variance = variance3.x;
}

void optimizer_sgd3(inout float3 primal, float3 grad, float learning_rate, bool renormalize)
{
    primal -= learning_rate * grad;

    if (renormalize)
        primal = normalize(primal);
    else    
        primal = saturate(primal);
}

void optimizer_sgd1(inout float primal, float grad, float learning_rate)
{
    primal -= learning_rate * grad;
    primal = saturate(primal);
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

