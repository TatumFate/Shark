#include "chassis.h"
#include "config.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

/* 每个编码器计数对应多少毫米车轮位移 */
#define MM_PER_TICK (M_PI * WHEEL_DIA_MM / (ENCODER_PPR * GEAR_RATIO * ENC_MULT))
#define DEG_PER_RAD 57.29578f
#define RAD_PER_DEG 0.017453293f

typedef enum { TASK_NONE = 0, TASK_TURN, TASK_DRIVE, TASK_GOTO, TASK_BLADE } TaskType;

static Pose g_pose;
static int32_t g_enc_l0, g_enc_r0;
static float g_v_l, g_v_r;
static float g_pwm_l, g_pwm_r;
static float g_i_l, g_i_r;
static float g_v_cmd, g_w_cmd;

static TaskType g_task;
static float g_t_h, g_t_x, g_t_y, g_t_dist;
static int   g_goto_phase;
static float g_start_x, g_start_y, g_start_h;
static int   g_blade_down;
static int32_t g_blade_ticks;

/* ---------------------------------------------------------------- 小工具 */
static float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float wrap180(float a)
{
    while (a > 180.0f)  a -= 360.0f;
    while (a < -180.0f) a += 360.0f;
    return a;
}

/* 单轮速度 PI + 前馈（1kHz） */
static float pid_step(float target, float actual, float *integ)
{
    const float dt = 1.0f / (float)CTRL_HZ;
    float err = target - actual;
    float ff  = PWM_MAX * target / MAX_SPEED_MM_S;
    *integ += err * dt;
    if (*integ >  PID_IMAX) *integ =  PID_IMAX;
    if (*integ < -PID_IMAX) *integ = -PID_IMAX;
    return clampf(ff + PID_KP * err + PID_KI * (*integ), -PWM_MAX, PWM_MAX);
}

/* ---------------------------------------------------------------- 初始化 */
void chassis_init(void)
{
    g_pose.x = 0.0f;
    g_pose.y = 0.0f;
    g_pose.h = 0.0f;
    g_pose.v = 0.0f;
    g_pose.w = 0.0f;
    hw_read_encoder(&g_enc_l0, &g_enc_r0);
    g_v_l = g_v_r = 0.0f;
    g_pwm_l = g_pwm_r = 0.0f;
    g_i_l = g_i_r = 0.0f;
    g_v_cmd = g_w_cmd = 0.0f;
    g_task = TASK_NONE;
    g_blade_down = 0;
    g_blade_ticks = 0;
    hw_set_pwm(0.0f, 0.0f);
    hw_set_blade(0);
}

/* ---------------------------------------------------------------- 指令 */
void chassis_cmd_goto(float x, float y, float h)
{
    g_t_x = x;
    g_t_y = y;
    g_t_h = wrap180(h);
    g_goto_phase = 0;              /* 0=转向目标点 1=直行 2=摆正 */
    g_i_l = g_i_r = 0.0f;
    g_task = TASK_GOTO;
}

void chassis_cmd_turn(float h)
{
    g_t_h = wrap180(h);
    g_i_l = g_i_r = 0.0f;
    g_task = TASK_TURN;
}

void chassis_cmd_drive(float dist_mm)
{
    g_t_dist = dist_mm;
    g_start_x = g_pose.x;
    g_start_y = g_pose.y;
    g_start_h = g_pose.h;
    g_i_l = g_i_r = 0.0f;
    g_task = TASK_DRIVE;
}

void chassis_cmd_blade(int down)
{
    g_blade_down = down ? 1 : 0;
    hw_set_blade(g_blade_down);
    g_blade_ticks = down ? BLADE_DOWN_MS : BLADE_UP_MS;
    g_task = TASK_BLADE;
}

void chassis_cmd_stop(void)
{
    g_task = TASK_NONE;
    g_v_cmd = g_w_cmd = 0.0f;
    g_i_l = g_i_r = 0.0f;
    hw_set_pwm(0.0f, 0.0f);
}

/* ---------------------------------------------------------------- 1kHz：里程计 + 速度环 */
void chassis_tick_fast(void)
{
    int32_t l, r;
    float dl, dr, ds, dh, mid;
    const float dt = 1.0f / (float)CTRL_HZ;
    float vl_t, vr_t;

    hw_read_encoder(&l, &r);
    dl = (float)(l - g_enc_l0) * (float)ENC_L_SIGN;
    dr = (float)(r - g_enc_r0) * (float)ENC_R_SIGN;
    g_enc_l0 = l;
    g_enc_r0 = r;

    g_v_l = dl * MM_PER_TICK / dt;
    g_v_r = dr * MM_PER_TICK / dt;

    /* 里程计：中值积分 */
    ds = 0.5f * (dl + dr) * MM_PER_TICK;
    dh = ((dl - dr) * MM_PER_TICK) / TRACK_MM;      /* 弧度，顺时针为正 */
    mid = g_pose.h * RAD_PER_DEG + dh * 0.5f;
    g_pose.x += ds * cosf(mid);
    g_pose.y += ds * sinf(mid);
    g_pose.h = wrap180(g_pose.h + dh * DEG_PER_RAD);
    g_pose.v = 0.5f * (g_v_l + g_v_r);
    g_pose.w = dh * DEG_PER_RAD / dt;

    /* 推板的"动作时间"用 1kHz 计数计时，时间到才算完成 */
    if (g_task == TASK_BLADE) {
        if (--g_blade_ticks <= 0) g_task = TASK_NONE;
    }

    /* 轮速目标：v 直行 + w 差速（w>0 顺时针，左轮走得多） */
    vl_t = g_v_cmd + g_w_cmd * RAD_PER_DEG * 0.5f * TRACK_MM;
    vr_t = g_v_cmd - g_w_cmd * RAD_PER_DEG * 0.5f * TRACK_MM;
    g_pwm_l = pid_step(vl_t, g_v_l, &g_i_l);
    g_pwm_r = pid_step(vr_t, g_v_r, &g_i_r);
    hw_set_pwm((float)MOTOR_L_SIGN * g_pwm_l, (float)MOTOR_R_SIGN * g_pwm_r);
}

/* ---------------------------------------------------------------- 50Hz：位置外环 */
void chassis_tick_outer(void)
{
    float v = 0.0f, w = 0.0f;

    switch (g_task) {
    case TASK_TURN: {
        float err = wrap180(g_t_h - g_pose.h);
        if (fabsf(err) < HEAD_TOL_DEG) { g_task = TASK_NONE; break; }
        w = clampf(err * 2.5f, -MAX_W_DEG_S, MAX_W_DEG_S);
        break;
    }
    case TASK_DRIVE: {
        float dx = g_pose.x - g_start_x;
        float dy = g_pose.y - g_start_y;
        float hr = g_start_h * RAD_PER_DEG;
        float travelled = dx * cosf(hr) + dy * sinf(hr);
        float left = g_t_dist - travelled;
        if (fabsf(left) < DRIVE_TOL_MM) { g_task = TASK_NONE; break; }
        v = clampf(left * 3.5f, -MAX_SPEED_MM_S, MAX_SPEED_MM_S);
        if (fabsf(v) < 150.0f) v = (left > 0.0f) ? 150.0f : -150.0f;
        break;
    }
    case TASK_GOTO: {
        float dx = g_t_x - g_pose.x;
        float dy = g_t_y - g_pose.y;
        float dist = sqrtf(dx * dx + dy * dy);
        if (g_goto_phase == 0) {
            float bearing = atan2f(dy, dx) * DEG_PER_RAD;
            float err = wrap180(bearing - g_pose.h);
            if (fabsf(err) < ARRIVE_TURN_TOL_DEG) { g_goto_phase = 1; break; }
            w = clampf(err * 2.5f, -MAX_W_DEG_S, MAX_W_DEG_S);
        } else if (g_goto_phase == 1) {
            float bearing, err;
            if (dist < POS_TOL_MM) { g_goto_phase = 2; break; }
            bearing = atan2f(dy, dx) * DEG_PER_RAD;
            err = wrap180(bearing - g_pose.h);
            w = clampf(err * 2.5f, -MAX_W_DEG_S, MAX_W_DEG_S);
            v = clampf(dist * 3.5f, 120.0f, MAX_SPEED_MM_S);
            v *= (1.0f - fabsf(err) / 120.0f);        /* 航向偏差大就减速 */
            if (v < 60.0f) v = 60.0f;
        } else {
            float err = wrap180(g_t_h - g_pose.h);
            if (fabsf(err) < HEAD_TOL_DEG) { g_task = TASK_NONE; break; }
            w = clampf(err * 2.5f, -MAX_W_DEG_S, MAX_W_DEG_S);
        }
        break;
    }
    default:
        break;
    }

    if (g_task == TASK_NONE) { v = 0.0f; w = 0.0f; }
    g_v_cmd = v;
    g_w_cmd = w;
}

/* ---------------------------------------------------------------- 查询 */
Pose chassis_pose(void)
{
    return g_pose;
}

int chassis_done(void)
{
    return (g_task == TASK_NONE) ? 1 : 0;
}

float chassis_pwm_l(void) { return g_pwm_l; }
float chassis_pwm_r(void) { return g_pwm_r; }
