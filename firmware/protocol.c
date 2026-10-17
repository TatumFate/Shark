#include "protocol.h"
#include "chassis.h"
#include <string.h>
#include <stdlib.h>

#define LINE_MAX 96

static char g_rx[LINE_MAX];
static volatile int g_len;
static char g_cmd[LINE_MAX];
static volatile int g_ready;
static uart_writer_t g_writer;

void proto_init(uart_writer_t writer)
{
    g_writer = writer;
    g_len = 0;
    g_ready = 0;
}

/* 在 UART 接收中断里调用。收到 \n 就把整行交给主循环。 */
void proto_rx_byte(uint8_t b)
{
    static int in_frame = 0;

    if (b == '$') {
        g_len = 0;
        in_frame = 1;
        return;
    }
    if (!in_frame) {
        return;
    }
    if (b == '\n' || b == '\r') {
        if (g_len > 0 && !g_ready) {
            memcpy(g_cmd, g_rx, (size_t)g_len);
            g_cmd[g_len] = 0;
            g_ready = 1;
        }
        g_len = 0;
        in_frame = 0;
        return;
    }
    if (g_len < LINE_MAX - 1) {
        g_rx[g_len++] = (char)b;
    }
}

/* 组帧：$body*XX\n   （XX = body 各字节求和取低 8 位，和上位机 Python 端一致） */
void proto_send_text(const char *body)
{
    char out[LINE_MAX + 8];
    const char *p;
    unsigned int sum = 0;
    unsigned int n = 0;
    static const char hex[] = "0123456789ABCDEF";

    if (g_writer == 0) {
        return;
    }
    for (p = body; *p != 0; p++) {
        sum += (unsigned char)(*p);
    }
    out[n++] = '$';
    for (p = body; *p != 0 && n < LINE_MAX; p++) {
        out[n++] = *p;
    }
    out[n++] = '*';
    out[n++] = hex[(sum >> 4) & 0x0F];
    out[n++] = hex[sum & 0x0F];
    out[n++] = '\n';
    g_writer(out, (int)n);
}

/* 把整数写进 dst，返回写入的字符数（不依赖 printf，省 flash 也省链接坑） */
static int fmt_int(char *dst, int v)
{
    char buf[12];
    int k = 0, n = 0, i;

    if (v < 0) { dst[n++] = '-'; v = -v; }
    if (v == 0) { buf[k++] = '0'; }
    while (v > 0) {
        buf[k++] = (char)('0' + v % 10);
        v /= 10;
    }
    for (i = k - 1; i >= 0; i--) {
        dst[n++] = buf[i];
    }
    return n;
}

void proto_send_pose(float x, float y, float h, int done)
{
    char body[LINE_MAX];
    int n = 0;
    int hi = (int)h;

    /* 只发整数：newlib-nano 默认不支持 %f，要额外链接选项，没必要踩这个坑 */
    while (hi > 180)  hi -= 360;
    while (hi < -180) hi += 360;
    body[n++] = 'P';
    body[n++] = ',';
    n += fmt_int(body + n, (int)x);
    body[n++] = ',';
    n += fmt_int(body + n, (int)y);
    body[n++] = ',';
    n += fmt_int(body + n, hi);
    body[n++] = ',';
    body[n++] = (char)(done ? '1' : '0');
    body[n] = 0;
    proto_send_text(body);
}

/* 解析并执行一条指令 */
static void handle_line(char *s)
{
    char *star;
    char *verb;
    unsigned int sum = 0;
    unsigned int got = 0;
    char *p;

    star = strchr(s, '*');
    if (star == 0) {
        return;
    }
    for (p = s; p < star; p++) {
        sum += (unsigned char)(*p);
    }
    got = (unsigned int)strtoul(star + 1, 0, 16);
    if ((sum & 0xFF) != (got & 0xFF)) {
        proto_send_text("ERR,CRC");
        return;
    }
    *star = 0;
    verb = strtok(s, ",");
    if (verb == 0) {
        return;
    }
    if (strcmp(verb, "GOTO") == 0) {
        char *a = strtok(0, ",");
        char *b = strtok(0, ",");
        char *c = strtok(0, ",");
        if (a && b && c) {
            chassis_cmd_goto((float)atof(a), (float)atof(b), (float)atof(c));
        }
    } else if (strcmp(verb, "TURN") == 0) {
        char *a = strtok(0, ",");
        if (a) chassis_cmd_turn((float)atof(a));
    } else if (strcmp(verb, "DRIVE") == 0) {
        char *a = strtok(0, ",");
        if (a) chassis_cmd_drive((float)atof(a));
    } else if (strcmp(verb, "BLADE") == 0) {
        char *a = strtok(0, ",");
        if (a) chassis_cmd_blade(atoi(a) ? 1 : 0);
    } else if (strcmp(verb, "STOP") == 0) {
        chassis_cmd_stop();
    } else if (strcmp(verb, "IDLE") == 0) {
        /* 什么都不做：保持当前状态 */
    } else {
        proto_send_text("ERR,VERB");
    }
}

void proto_poll(void)
{
    char local[LINE_MAX];

    if (!g_ready) {
        return;
    }
    memcpy(local, g_cmd, LINE_MAX);
    g_ready = 0;                    /* 先释放缓冲，解析期间可以继续收 */
    handle_line(local);
}
