#ifndef CHASSIS_H
#define CHASSIS_H

#include <stdint.h>

typedef struct {
    float x;        /* mm */
    float y;        /* mm */
    float h;        /* 度，0=+x，顺时针为正 */
    float v;        /* mm/s */
    float w;        /* 度/s */
} Pose;

/* 初始化：清零里程计、任务、PID */
void chassis_init(void);

/* ---- 指令（由 protocol 层调用）---- */
void chassis_cmd_goto(float x, float y, float h);
void chassis_cmd_turn(float h);
void chassis_cmd_drive(float dist_mm);     /* 负数=倒车 */
void chassis_cmd_blade(int down);
void chassis_cmd_stop(void);

/* ---- 周期性调用 ---- */
void chassis_tick_fast(void);      /* 1kHz：读编码器 + 里程计 + 轮速 PID */
void chassis_tick_outer(void);     /* 50Hz：位置外环（GOTO/TURN/DRIVE） */

/* ---- 供上报 / 调试 ---- */
Pose  chassis_pose(void);
int   chassis_done(void);          /* 1=当前指令已完成 */
float chassis_pwm_l(void);         /* 调试用 */
float chassis_pwm_r(void);

/* ---- 硬件钩子：在 app_main.c 里实现 ---- */
void  hw_read_encoder(int32_t *left, int32_t *right);   /* 读计数器累计值 */
void  hw_set_pwm(float left, float right);              /* -PWM_MAX..+PWM_MAX */
void  hw_set_blade(int down);                           /* 1=放下 */

#endif /* CHASSIS_H */
