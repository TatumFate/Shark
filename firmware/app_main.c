/* ==========================================================================
 *  app_main.c —— STM32 主程序骨架
 *
 *  ⚠ 这是**未编译验证**的骨架：请并进你们自己的 CubeMX 工程（STM32F407），
 *    按实际引脚改 hw_* 三个函数和下面的 extern 句柄名。
 *
 *  分工：上位机（Python）负责"看 + 决策"，发 $GOTO/$TURN/$DRIVE/$BLADE/$STOP；
 *        本文件负责"执行"：解析指令 → 差速运动学 → 轮速 PI → 里程计 → 回传位姿。
 *
 *  上电顺序（别跳步，见 firmware/README.md）：
 *    1. 只让轮子空转，确认方向对
 *    2. 看里程计：手推 1m，看 x 是否 ≈1000mm
 *    3. 原地转 90°，看 h 是否 ≈90
 *    4. 串口回环：电脑发 $STOP*..，板子回 $P,...
 *    5. 接视觉，先跑 $GOTO 到一个点，再跑整局
 * ========================================================================== */

#include "main.h"
#include "chassis.h"
#include "protocol.h"
#include "config.h"

/* ---- CubeMX 生成的句柄：名字按你们的工程改 ---- */
extern TIM_HandleTypeDef htim1;      /* 编码器左（Encoder Mode） */
extern TIM_HandleTypeDef htim2;      /* 编码器右（Encoder Mode） */
extern TIM_HandleTypeDef htim3;      /* PWM：CH1=左轮，CH2=右轮 */
extern TIM_HandleTypeDef htim6;      /* 1kHz 基础定时器（速度环时基） */
extern UART_HandleTypeDef huart2;    /* 与上位机通信 */

static uint8_t g_rx_byte;

/* ==========================================================================
 *  硬件钩子
 * ========================================================================== */
void hw_read_encoder(int32_t *left, int32_t *right)
{
    /* 16 位计数器会回绕，用 (int16_t) 差值累加，天然处理回绕 */
    static uint16_t pl = 0, pr = 0;
    static int32_t acc_l = 0, acc_r = 0;
    uint16_t cl = (uint16_t)__HAL_TIM_GET_COUNTER(&htim1);
    uint16_t cr = (uint16_t)__HAL_TIM_GET_COUNTER(&htim2);

    acc_l += (int32_t)(int16_t)(cl - pl);
    acc_r += (int32_t)(int16_t)(cr - pr);
    pl = cl;
    pr = cr;
    *left = acc_l;
    *right = acc_r;
}

void hw_set_pwm(float left, float right)
{
    /* TODO: 按你们的驱动改引脚。这里假设：PWM 通道 + 一个 DIR 脚控制正反转 */
    int pwm_l = (int)left;
    int pwm_r = (int)right;

    if (pwm_l > (int)PWM_MAX) pwm_l = (int)PWM_MAX;
    if (pwm_l < -(int)PWM_MAX) pwm_l = -(int)PWM_MAX;
    if (pwm_r > (int)PWM_MAX) pwm_r = (int)PWM_MAX;
    if (pwm_r < -(int)PWM_MAX) pwm_r = -(int)PWM_MAX;

    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_0, (pwm_l >= 0) ? GPIO_PIN_SET : GPIO_PIN_RESET);
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1, (pwm_r >= 0) ? GPIO_PIN_SET : GPIO_PIN_RESET);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, (uint32_t)(pwm_l >= 0 ? pwm_l : -pwm_l));
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, (uint32_t)(pwm_r >= 0 ? pwm_r : -pwm_r));
}

void hw_set_blade(int down)
{
    /* TODO: 换成你们的舵机/减速电机/电磁离合。这里给的是 GPIO 控制继电器 */
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_7, down ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/* ==========================================================================
 *  UART 发送钩子
 * ========================================================================== */
static void uart_write(const char *s, int len)
{
    HAL_UART_Transmit(&huart2, (uint8_t *)s, (uint16_t)len, 20);
}

/* ==========================================================================
 *  主程序
 * ========================================================================== */
int main(void)
{
    uint32_t t_outer = 0, t_telem = 0;

    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_TIM1_Init();
    MX_TIM2_Init();
    MX_TIM3_Init();
    MX_TIM6_Init();
    MX_USART2_UART_Init();

    HAL_TIM_Encoder_Start(&htim1, TIM_CHANNEL_ALL);
    HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
    HAL_TIM_Base_Start_IT(&htim6);            /* 1kHz 中断里跑速度环 */
    HAL_UART_Receive_IT(&huart2, &g_rx_byte, 1);

    proto_init(uart_write);
    chassis_init();
    proto_send_text("BOOT,OK");

    while (1) {
        uint32_t now = HAL_GetTick();

        if ((uint32_t)(now - t_outer) >= (1000u / OUTER_HZ)) {
            t_outer = now;
            chassis_tick_outer();             /* 位置外环 50Hz */
        }
        if ((uint32_t)(now - t_telem) >= (1000u / TELEM_HZ)) {
            Pose p;
            t_telem = now;
            p = chassis_pose();
            proto_send_pose(p.x, p.y, p.h, chassis_done());   /* 25Hz 回传位姿 */
        }
        proto_poll();                         /* 解析一条指令 */
    }
}

/* ==========================================================================
 *  中断回调
 * ========================================================================== */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6) {
        chassis_tick_fast();                  /* 1kHz：编码器 + 里程计 + 轮速 PI */
    }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2) {
        proto_rx_byte(g_rx_byte);
        HAL_UART_Receive_IT(&huart2, &g_rx_byte, 1);
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2) {
        /* 溢出/噪声错误后必须重启接收，否则之后再也收不到指令 */
        HAL_UART_Receive_IT(&huart2, &g_rx_byte, 1);
    }
}
