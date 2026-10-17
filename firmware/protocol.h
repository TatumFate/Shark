#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

/* 把数据发出去的钩子（app_main 里用 HAL_UART_Transmit 实现） */
typedef void (*uart_writer_t)(const char *s, int len);

void proto_init(uart_writer_t writer);
void proto_rx_byte(uint8_t b);       /* UART 接收中断里逐字节调用 */
void proto_poll(void);               /* 主循环里调用：解析并执行一条完整指令 */
void proto_send_pose(float x, float y, float h, int done);
void proto_send_text(const char *body);

#endif /* PROTOCOL_H */
