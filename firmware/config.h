#ifndef CONFIG_H
#define CONFIG_H

/* ==========================================================================
 *  机械与控制参数：全部按实物量出来再填
 *  （单位统一 mm / s / 度；角度约定：0°=+x，顺时针为正，与上位机一致）
 * ========================================================================== */

/* ---- 机械 ---- */
#define WHEEL_DIA_MM      65.0f     /* 驱动轮直径 */
#define TRACK_MM         180.0f     /* 两驱动轮中心距 */
#define ENCODER_PPR       11.0f     /* 编码器每圈脉冲（单相，查电机手册） */
#define GEAR_RATIO        30.0f     /* 减速比 */
#define ENC_MULT           4.0f     /* 编码器计数倍频：定时器编码器模式通常是 4 倍频 */

/* ---- 方向：第一次上电必测！转反了就把对应的 SIGN 改成 -1 ---- */
#define MOTOR_L_SIGN      (+1)
#define MOTOR_R_SIGN      (+1)
#define ENC_L_SIGN        (+1)
#define ENC_R_SIGN        (+1)

/* ---- 速度环 ---- */
#define CTRL_HZ           1000      /* 速度环频率（1kHz，在定时器中断里调用） */
#define OUTER_HZ            50      /* 位置外环频率 */
#define TELEM_HZ            25      /* 位姿回传频率 */
#define PWM_MAX           1000.0f   /* 定时器 ARR 对应 1000 */
#define MAX_SPEED_MM_S    1200.0f
#define MAX_W_DEG_S        120.0f
#define PID_KP              1.6f
#define PID_KI              3.0f
#define PID_IMAX         8000.0f

/* ---- 位置控制 ---- */
#define POS_TOL_MM          25.0f   /* GOTO 到位判定 */
#define HEAD_TOL_DEG         3.0f   /* 摆正到位判定 */
#define ARRIVE_TURN_TOL_DEG  6.0f   /* 去目标点前先转到的容差 */
#define DRIVE_TOL_MM         8.0f   /* DRIVE 到位判定 */

/* ---- 场地（和上位机 config.py 保持一致）---- */
#define FIELD_L_MM        3000.0f
#define FIELD_W_MM        2000.0f

/* ---- 推板机构：按实际执行器改，这里给的是"一个 GPIO 控制继电器/舵机"的占位 ---- */
#define BLADE_DOWN_MS      600      /* 放下推板需要的时间 */
#define BLADE_UP_MS        500      /* 抬起推板需要的时间 */

#endif /* CONFIG_H */
