// Modified by Jinchen on 2024-3-17 on unitarity check

/* Only Nc=3 reader is supported */
#include "modules.h"                                                 /* DEPS */
#include "qlua.h"                                                    /* DEPS */
#include "lattice.h"                                                 /* DEPS */
#include "qlayout.h"                                                 /* DEPS */
#include "latcolmat.h"                                               /* DEPS */
#include "qend.h"                                                    /* DEPS */
#include "qmp.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <mpi.h>

#if USE_Nc3
static const char nersc_io[] = "nersc";

static int
nersc_error(lua_State *L, const char *errmsg)
{
    return luaL_error(L, "L:read_NERSC_gauge() error: %s", errmsg);
}

static char *
nersc_hdrnorm(char *buffer)
{
    char *head, *tail;

    head = buffer;
    tail = buffer + strlen(buffer) - 1;
    
    while (*head == ' ')
        ++head;
    while (*tail == ' ')
        *tail-- = 0;

    return head;
}

#define NERSC_BUFSIZE 1024

static int
nersc_gethdr(FILE *f, char *buffer, char **key, char **value)
{
    char *eq;

    if (fgets(buffer, NERSC_BUFSIZE, f) == 0)
        return 0;
    buffer[NERSC_BUFSIZE - 1] = 0;
    eq = buffer + strlen(buffer) - 1;
    if (*eq != '\n')
        return 0;
    *eq = 0;

    eq = strchr(buffer, '=');
    if (eq == NULL) {
        *key = nersc_hdrnorm(buffer);
        *value = NULL;
        return 1;
    } else {
        *eq++ = 0;
        *key = nersc_hdrnorm(buffer);
        *value = nersc_hdrnorm(eq);
        return 2;
    }
}
static int
nersc_puthdr(FILE *f, char *buffer, const char **key, const char **value)
{
    char *eq;

    if (fgets(buffer, NERSC_BUFSIZE, f) == 0)
        return 0;
    buffer[NERSC_BUFSIZE - 1] = 0;
    eq = buffer + strlen(buffer) - 1;
    if (*eq != '\n')
        return 0;
    *eq = 0;

    eq = strchr(buffer, '=');
    if (eq == NULL) {
        *key = nersc_hdrnorm(buffer);
        *value = NULL;
        return 1;
    } else {
        *eq++ = 0;
        *key = nersc_hdrnorm(buffer);
        *value = nersc_hdrnorm(eq);
        return 2;
    }
}

typedef struct {
    const char *name;
    int value;
} NERSC_Value;

static int
decode_hdr(lua_State *L, const char *value, int old, int expected,
           const NERSC_Value *t, char *msg, char **status)
{
    int i;

    if (*status != NULL)
        return old;

    if (old != expected) {
        *status = msg;
        return old;
    }

    for (i = 0; t[i].name; i++) {
        if (strcmp(t[i].name, value) == 0)
            return t[i].value;
    }
    *status = msg;
    return old;
}

typedef double (*RealReader)(const char *restrict data, int idx);
static double
read_float(const char *restrict data, int idx)
{
    return *(const float *)(data + idx * sizeof (float));
}

static double
read_double(const char *restrict data, int idx)
{
    return *(const double *)(data + idx * sizeof (double));
}

// FIXME can replace nd -> S->rank
typedef void (*GaugeReader)(lua_State *L,
                            mLattice *S,
                            QLA_D3_ColorMatrix *restrict U, int nd,
                            const char *restrict buf, int buf_size,
                            RealReader read_real);

static void
read_3x3(lua_State *L,
         mLattice *S,
         QLA_D3_ColorMatrix *restrict U, int nd,
         const char *restrict buf, int buf_size,
         RealReader read_real)
{
    int i, d, a, b;

    for (i = 0, d = 0; d < nd; d++) {
        for (a = 0; a < 3; a++) {
            for (b = 0; b < 3; b++, i += 2) {
                QLA_c_eq_r_plus_ir(QLA_D3_elem_M(U[d], a, b),
                                   read_real(buf, i),
                                   read_real(buf, i + 1));
            }
        }
    }
}

static void
read_3x2(lua_State *L,
         mLattice *S,
         QLA_D3_ColorMatrix *restrict U, int nd,
         const char *restrict buf, int buf_size,
         RealReader read_real)
{
    int i, d, a, b;

    for (i = 0, d = 0; d < nd; d++, U++) {
        for (a = 0; a < 3 - 1; a++) {
            for (b = 0; b < 3; b++, i += 2) {
                QLA_c_eq_r_plus_ir(QLA_D3_elem_M(*U, a, b),
                                   read_real(buf, i),
                                   read_real(buf, i + 1));
            }
        }
        QLA_c_eq_ca_times_ca( QLA_D3_elem_M(*U,2,2), QLA_D3_elem_M(*U,0,0),
                              QLA_D3_elem_M(*U,1,1));
        QLA_c_meq_ca_times_ca(QLA_D3_elem_M(*U,2,2), QLA_D3_elem_M(*U,0,1),
                              QLA_D3_elem_M(*U,1,0));

        QLA_c_eq_ca_times_ca( QLA_D3_elem_M(*U,2,1), QLA_D3_elem_M(*U,0,2),
                              QLA_D3_elem_M(*U,1,0));
        QLA_c_meq_ca_times_ca(QLA_D3_elem_M(*U,2,1), QLA_D3_elem_M(*U,0,0),
                              QLA_D3_elem_M(*U,1,2));

        QLA_c_eq_ca_times_ca( QLA_D3_elem_M(*U,2,0), QLA_D3_elem_M(*U,0,1),
                              QLA_D3_elem_M(*U,1,2));
        QLA_c_meq_ca_times_ca(QLA_D3_elem_M(*U,2,0), QLA_D3_elem_M(*U,0,2),
                              QLA_D3_elem_M(*U,1,1));
    }
}

typedef void (*RealWriter)(char *restrict data, int idx, double v);
static void
write_float(char *restrict data, int idx, double v)
{
    ((float *)data)[idx]    = v;
}

static void
write_double(char *restrict data, int idx, double v)
{
    ((double *)data)[idx]   = v;
}

typedef void (*GaugeWriter)(
        lua_State *L, mLattice *S, char *restrict buf,
        const QLA_D3_ColorMatrix *restrict U, RealWriter write_real);
static void
write_3x3(
        lua_State *L, mLattice *S, char *restrict buf,
         const QLA_D3_ColorMatrix *restrict U, RealWriter write_real)
{
    int i, d, a, b;

    for (i = 0, d = 0; d < S->rank; d++) {
        for (a = 0; a < 3; a++) {
            for (b = 0; b < 3; b++, i += 2) {
                write_real(buf, i    , QLA_real(QLA_D3_elem_M(U[d], a, b)));
                write_real(buf, i + 1, QLA_imag(QLA_D3_elem_M(U[d], a, b)));
            }
        }
    }
}

static void
write_3x2(
        lua_State *L, mLattice *S, char *restrict buf,
        const QLA_D3_ColorMatrix *restrict U, RealWriter write_real)
{
    int i, d, a, b;

    for (i = 0, d = 0; d < S->rank; d++) {
        for (a = 0; a < 3 - 1; a++) {
            for (b = 0; b < 3; b++, i += 2) {
                write_real(buf, i    , QLA_real(QLA_D3_elem_M(U[d], a, b)));
                write_real(buf, i + 1, QLA_imag(QLA_D3_elem_M(U[d], a, b)));
            }
        }
    }
}

static void
nersc_site2coord(int *coord, long long site, int nd, const int *dim)
{
    int i;

    for (i = 0; i < nd; i++) {
        coord[i] = site % dim[i];
        site = site / dim[i];
    }
}
static long long 
nersc_coord2site(const int *coord, int nd, const int *dim)
{
    int i, res = 0;

    for (i = nd; i-- ;) {
        res = res * dim[i] + coord[i];
    }
    return res;
}

static void
normalize_int(lua_State *L, int idx, const char *key, char *fmt)
{
    lua_getfield(L, idx, key);
    const char *value = lua_tostring(L, -1);
    int v;
    if (value == NULL)
        return;

    if (sscanf(value, fmt, &v) == 1) {
        lua_pop(L, 1);
        lua_pushnumber(L, v);
        lua_setfield(L, idx - 1, key);
        return;
    }
    lua_pop(L, 1);
}

static void
normalize_float(lua_State *L, int idx, const char *key)
{
    lua_getfield(L, idx, key);
    const char *value = lua_tostring(L, -1);
    double v;
    if (value == NULL)
        return;
    if (sscanf(value, "%lg", &v) == 1) {
        lua_pop(L, 1);
        lua_pushnumber(L, v);
        lua_setfield(L, idx - 1, key);
        return;
    }
    lua_pop(L, 1);
}

static const char *ukey = "unitarity";

static void
normalize_kv(lua_State *L, mLattice *S, int idx)
{
    int i;
    for (i = 1; i <= S->rank; i++) {
        char buf[128]; /* large enough for DIMENSION_%d */
        snprintf(buf, sizeof (buf) - 1, "DIMENSION_%d", i);
        normalize_int(L, idx, buf, "%d");
    }
    normalize_int(L, idx, "CHECKSUM", "%x");
    normalize_float(L, idx, "PLAQUETTE");
    normalize_float(L, idx, "LINK_TRACE");
    normalize_float(L, idx, ukey);
}

static void
default_setup(lua_State *L)
{
    switch (lua_gettop(L)) {
    case 2:
        /* create an empty table at #3 */
        lua_createtable(L, 0, 0);
        break;
    case 3:
        if (!lua_istable(L, 3))
            luaL_error(L, "expecting an overwrite table");
        break;
    default:
        luaL_error(L, "wrong number of arguments");
        break;
    }
}

static double
default_double(lua_State *L, const char *key, double def)
{
    double v = def;

    lua_pushstring(L, key);
    lua_gettable(L, 3);
    if (lua_isnil(L, -1)) {
        v = def;
    } else {
        v = luaL_checknumber(L, -1);
    }
    lua_pop(L, 1);
    return v;
}

static int
default_enum(lua_State *L, const char *key, int def, const NERSC_Value *tr)
{
    int v = def;

    lua_pushstring(L, key);
    lua_gettable(L, 3);
    if (lua_isnil(L, -1)) {
        v = def;
    } else {
        const char *val = luaL_checkstring(L, -1);
        
        for (; tr->name; tr++) {
            if (strcmp(tr->name, val) == 0) {
                v = tr->value;
                break;
            }
        }
        if (tr->name == NULL)
            luaL_error(L, "unexpected overwrite value");
    }
    lua_pop(L, 1);
    return v;
}

static int
nersc_read_master(lua_State *L,
                  mLattice *S,
                  QLA_D3_ColorMatrix **U,
                  const char *name)
{
    enum {
        ntNONE,
        nt4D_3x3,
        nt4D_3x2
    };
    static const NERSC_Value nFMTs[] = {
        {"4D_SU3_GAUGE_3x3", nt4D_3x3 },
        {"4D_SU3_GAUGE",     nt4D_3x2 },
        {NULL, ntNONE}
    };
    static const NERSC_Value nFPs[] = {
        {"IEEE32",           4},
        {"IEEE32BIG",        4},
        {"IEEE64BIG",        8},
        {"IEEE32LITTLE",    16},
        {"IEEE64LITTLE",    32},
        {NULL,               0}
    };

    FILE *f = fopen(name, "rb");
    char buffer[NERSC_BUFSIZE];
    char *key, *value;
    int  *coord = qlua_malloc(L, S->rank * sizeof (int));
    char *dim_ok = qlua_malloc(L, S->rank * sizeof (int));
    long long volume;
    long long site;
    int s_node;
    int i;
    int f_format = ntNONE;
    int f_fp = 0;
    int f_cs_p = 0;
    uint32_t f_checksum = 0;
    uint32_t d_checksum = 0;
    GaugeReader read_matrix = NULL;
    RealReader read_real = NULL;
    int site_size = 0;
    int big_endian;
    double uni_eps = 0;
    QLA_D3_ColorMatrix mone;
    QLA_D_Complex cone;
    double max_eps = 0.0;
    char *status = NULL;

    QLA_c_eq_r_plus_ir(cone, 1.0, 0.0);
    QLA_D3_M_eq_c(&mone, &cone);

    if (f == 0)
        status = "file open error";

    for (i = 0; i < S->rank; i++)
        dim_ok[i] = 0;

    /* parse the header */
    if ((status == NULL) && 
        ((nersc_gethdr(f, buffer, &key, &value) != 1) ||
         (strcmp(key, "BEGIN_HEADER") != 0)))
        status = "missing header";

    while (status == NULL) {
        switch (nersc_gethdr(f, buffer, &key, &value)) {
        case 1:
            if (strcmp(key, "END_HEADER") != 0)
                status = "missing end of header";
            goto eoh;
        case 2:
            lua_pushstring(L, value);
            lua_setfield(L, -2, key);
            
            if (strcmp(key, "DATATYPE") == 0) {
                f_format = decode_hdr(L, value, f_format, ntNONE, nFMTs,
                                      "bad or conflicting DATATYPE", &status);
            } else if (strcmp(key, "FLOATING_POINT") == 0) {
                f_fp = decode_hdr(L, value, f_fp, 0, nFPs,
                                  "bad or conflicting FLOATING_POINT", &status);
            } else if (strcmp(key, "CHECKSUM") == 0) {
                if (f_cs_p)
                    status = "multiple CHECKSUMs";
                if ((status == NULL) && 
                    (sscanf(value, "%x", &f_checksum) != 1))
                    status = "illformed CHECKSUM";
                f_cs_p = 1;
            } else if (sscanf(key, "DIMENSION_%d", &i) == 1) {
                int di;
                if ((i < 1) || (i > S->rank))
                    status = "DIMENSION out of range";
                if ((status == NULL) &&
                    ((sscanf(value, "%d", &di) != 1) ||
                     (S->dim[i - 1] != di)))
                    status = "DIMENSION mismatch";
                dim_ok[i - 1] = 1;
            } else if (sscanf(key, "BOUNDARY_%d", &i) == 1) {
                if ((i < 1) || (i > S->rank))
                    status = "BOUNDARY out of range";
                if ((status == NULL) &&
                    (strcmp(value, "PERIODIC") != 0))
                    status = "bad BOUNDARY value";
            } else if ((strcmp(key, "PLAQUETTE") == 0) ||
                       (strcmp(key, "LINK_TRACE") == 0)) {
                double v;
                if (sscanf(value, "%lg", &v) != 1)
                    status = "unexpected value";
            }
            break;
        default:
            status = "illformed header";
        }
    }
eoh:
    /* get defaults from the call */
    f_format = default_enum(L, "DATATYPE", f_format, nFMTs);
    f_fp = default_enum(L, "FLOATING_POINT", f_fp, nFPs);
    uni_eps = default_double(L, ukey, 0.0);

    switch (f_format) {
    case nt4D_3x3:
        read_matrix = read_3x3;
        site_size = S->rank * 3 * 3 * 2;
        break;
    case nt4D_3x2:
        read_matrix = read_3x2;
        site_size = S->rank * 3 * (3 - 1) * 2;
        break;
    default:
        if (status == NULL)
            status = "unsupported data format";
    }
    int big_endian_data = 1;    /* default value */
    switch (f_fp) {
    case 4:
        read_real = read_float;
        site_size *= 4;
        big_endian_data = 1;
        if (uni_eps == 0) uni_eps = 1e-6;
        break;
    case 8:
        read_real = read_double;
        site_size *= 8;
        big_endian_data = 1;
        if (uni_eps == 0) uni_eps = 1e-12;
        break;
    case 16:
        read_real = read_float;
        site_size *= 4;
        big_endian_data = 0;
        if (uni_eps == 0) uni_eps = 1e-6;
        break;
    case 32:
        read_real = read_double;
        site_size *= 8;
        big_endian_data = 0;
        if (uni_eps == 0) uni_eps = 1e-12;
        break;

    default:
        if (status == NULL)
            status = "bad floating point size";
    }
    for (i = 0; i < S->rank; i++) {
        if ((!dim_ok[i]) && (status == NULL)) {
            status = "missing DIMENSION spec";
        }
    }
    if ((f_cs_p == 0) && (status == NULL))
        status = "missing CHECKSUM";

    /* Read the data and send it to the target host */
    for (volume = 1, i = 0; i < S->rank; i++)
        volume *= S->dim[i];
    /* find out our endianess */
    /* FIXME replace with std function */
    {
        union {
            uint64_t ll;
            unsigned char c[sizeof (uint64_t)];
        } b;
        uint64_t v;
        big_endian = 1;
        for (v = 1, i = 0; i < (long long)sizeof (uint64_t); i++)
            v = (v << CHAR_BIT) + i + 1;
        b.ll = v;
        if (b.c[0] == 1)
            big_endian = 1;
        else if (b.c[0] == i)
            big_endian = 0;
        else if (status != NULL)
            status = "Unexpected host endianness";
    }
    /* read every site in order on the master
     * compute the checksum and send it to the target node
     */
    char *site_buf = qlua_malloc(L, site_size);
    QLA_D3_ColorMatrix *CM = qlua_malloc(L, S->rank * sizeof (QLA_D3_ColorMatrix));
    QMP_msgmem_t mm = QMP_declare_msgmem(&CM[0], S->rank * sizeof (CM[0]));

    /* Go through all sites */
    for (site = 0; site < volume; site++) {
        if ((status == NULL) && (fread(site_buf, site_size, 1, f) != 1))
            status = "file read error";

        /* swap bytes if necessary */
        if ((big_endian && !big_endian_data) 
                || (!big_endian && big_endian_data)) {
            char *p;
            switch (f_fp) {
            case 4:
                for (i = 0, p = site_buf; i < site_size; i += 4, p += 4) {
                    char t;
                    t = p[0]; p[0] = p[3]; p[3] = t;
                    t = p[1]; p[1] = p[2]; p[2] = t;
                }
                break;
            case 8:
                for (i = 0, p = site_buf; i < site_size; i += 8, p += 8) {
                    char t;
                    t = p[0]; p[0] = p[7]; p[7] = t;
                    t = p[1]; p[1] = p[6]; p[6] = t;
                    t = p[2]; p[2] = p[5]; p[5] = t;
                    t = p[3]; p[3] = p[4]; p[4] = t;
                }
                break;
            default:
                if (status == NULL) 
                    status = "internal error: unsupported f_fp in endiannes conversion";
            }
        }
        /* collect the checksum */
        for (i = 0; i < site_size; i += sizeof (uint32_t))
            d_checksum += *(uint32_t *)(site_buf + i);
        /* convert to the ColorMatrix */
        if (read_matrix != NULL && read_real != NULL) 
            read_matrix(L, S, CM, S->rank, site_buf, site_size, read_real); 
        /* check unitarity */
        // {
        //     int d, a, b;
        //     QLA_D3_ColorMatrix UxU;

        //     for (d = 0; d < S->rank; d++) {
        //         double er, ei;

        //         /* multiplcation order helps to detect reconstruction bugs */
        //         QLA_D3_M_eq_M_times_Ma(&UxU, &CM[d], &CM[d]);
        //         QLA_D3_M_meq_M(&UxU, &mone);
        //         for (a = 0; a < 3; a++) {
        //             for (b = 0; b < 3; b++) {
        //                 QLA_Complex z;

        //                 QLA_D3_C_eq_elem_M(&z, &UxU, a, b);
        //                 er = fabs(QLA_real(z));
        //                 ei = fabs(QLA_imag(z));
        //                 if (((er > uni_eps) || (ei > uni_eps)) && (status == NULL))
        //                     status = "unitarity violation";
        //                 if (er > max_eps)
        //                     max_eps = er;
        //                 if (ei > max_eps)
        //                     max_eps = ei;
        //             }
        //         }
        //     }
        // }
        /* check unitarity modified by Jinchen on 2024-3-17 */
        {
            int d, a, b;
            QLA_D3_ColorMatrix UxU;
            double max_deviation = 0.0; // new variable

            for (d = 0; d < S->rank; d++) {
                double er, ei;

                /* multiplcation order helps to detect reconstruction bugs */
                QLA_D3_M_eq_M_times_Ma(&UxU, &CM[d], &CM[d]);
                QLA_D3_M_meq_M(&UxU, &mone);
                for (a = 0; a < 3; a++) {
                    for (b = 0; b < 3; b++) {
                        QLA_Complex z;

                        QLA_D3_C_eq_elem_M(&z, &UxU, a, b);
                        er = fabs(QLA_real(z));
                        ei = fabs(QLA_imag(z));
                        if (er > max_eps)
                            max_eps = er;
                        if (ei > max_eps)
                            max_eps = ei;
                        // renew the max deviation
                        if (max_deviation < er) max_deviation = er;
                        if (max_deviation < ei) max_deviation = ei;
                    }
                }
            }
            // check and print out the deviation, instead of stop
            if (max_deviation > uni_eps) {
                printf("Unitarity violation detected. Max deviation: %e\n", max_deviation);
            }
        }
        /* place ColorMatrix in U where it belongs */
        nersc_site2coord(coord, site, S->rank, S->dim);
        s_node = QDP_node_number_L(S->lat, coord);
        if (s_node == QDP_this_node) {
            int idx = QDP_index_L(S->lat, coord);
            int d;
            
            for (d = 0; d < S->rank; d++)
                QLA_D3_M_eq_M(&U[d][idx], &CM[d]);
        } else {
            QMP_msghandle_t mh = QMP_declare_send_to(mm, s_node, 0);
            QMP_start(mh);
            QMP_wait(mh);
            QMP_free_msghandle(mh);
        }
    }
    QMP_free_msgmem(mm);
    qlua_free(L, coord);
    qlua_free(L, dim_ok);
    qlua_free(L, site_buf);
    qlua_free(L, CM);
        
    if (f)
      fclose(f);

    /* check the checksum */
    if ((status == NULL) && (f_checksum != d_checksum))
        status = "checksum mismatch";

    /* broadcast the size of status to everyone */
    int status_len = (status != NULL)? (strlen(status) + 1): 0;
    QMP_sum_int(&status_len);
    /* if status is not NULL, broadcast to to everyone and call lua error */
    if (status_len != 0) {
        QMP_broadcast(status, status_len);
        return nersc_error(L, status);
    }

    /* convert max_eps to a string and store it (L, -2, ukey) */
    snprintf(buffer, sizeof (buffer) - 1, "%.10e", max_eps);
    lua_pushstring(L, buffer);
    lua_setfield(L, -2, ukey);
    /* ditribute the table across the machine */
    {
        lua_pushnil(L);
        while (lua_next(L, -2) != 0) {
            qlua_send_string(L, -2); /* key */
            qlua_send_string(L, -1); /* value */
            lua_pop(L, 1);
        }
        int zero = 0;
        QMP_broadcast(&zero, sizeof (zero));
    }
    /* normalize key/value pairs */
    normalize_kv(L, S, -1);
    return 2;
}

static int
nersc_read_slave(lua_State *L,
                 mLattice *S,
                 QLA_D3_ColorMatrix **U,
                 const char *name)
{
    long long volume;
    long long site;
    int i;
    QLA_D3_ColorMatrix *CM = qlua_malloc(L, S->rank * sizeof (QLA_D3_ColorMatrix));
    int *coord = qlua_malloc(L, S->rank * sizeof (int));
    QMP_msgmem_t mm = QMP_declare_msgmem(&CM[0], S->rank * sizeof (CM[0]));

    /* get gauge element for this node */
    for (volume = 1, i = 0; i < S->rank; i++)
        volume *= S->dim[i];

    /* get all data from node qlua_master_node in nersc order */
    for (site = 0; site < volume; site++) {
        int s_node;
        nersc_site2coord(coord, site, S->rank, S->dim);
        s_node = QDP_node_number_L(S->lat, coord);
        if (s_node == QDP_this_node) {
            QMP_msghandle_t mh = QMP_declare_receive_from(mm, qlua_master_node, 0);
            QMP_start(mh);
            QMP_wait(mh);
            QMP_free_msghandle(mh);

            int idx = QDP_index_L(S->lat, coord);
            int d;
            
            for (d = 0; d < S->rank; d++)
                QLA_D3_M_eq_M(&U[d][idx], &CM[d]);
        }
    }
    QMP_free_msgmem(mm);
    qlua_free(L, coord);
    qlua_free(L, CM);

    /* get status size (broadcast) */
    int status_len = 0;
    QMP_sum_int(&status_len);
    /* if not zero, get the message, call lua_error */
    if (status_len != 0) {
        char *msg = qlua_malloc(L, status_len);
        QMP_broadcast(msg, status_len);
        return nersc_error(L, msg); /* leaks msg, but it's in the abort path only */
    }
        
    /* get keys and values */
    /* NB: This may cause a slave node to run out of memory out of sync with the master */
    for (;;) {
        char *key, *value;
        if (qlua_receive_string(L, &key) == 0)
            break;
        qlua_receive_string(L, &value);
        lua_pushstring(L, value);
        lua_setfield(L, -2, key);
        qlua_free(L, key);
        qlua_free(L, value);
    }
        
    /* normalize key/value table */
    normalize_kv(L, S, -1);
    return 2;
}

static int
nersc_read_parallel(lua_State *L,
                  mLattice *S,
                  QLA_D3_ColorMatrix **U,
                  const char *name)
{
    char *status = NULL;
    enum {
        ntNONE,
        nt4D_3x3,
        nt4D_3x2
    };
    static const NERSC_Value nFMTs[] = {
        {"4D_SU3_GAUGE_3x3", nt4D_3x3 },
        {"4D_SU3_GAUGE",     nt4D_3x2 },
        {NULL, ntNONE}
    };
    static const NERSC_Value nFPs[] = {
        {"IEEE32",           4},
        {"IEEE32BIG",        4},
        {"IEEE64BIG",        8},
        {"IEEE32LITTLE",    16},
        {"IEEE64LITTLE",    32},
        {NULL,               0}
    };

    FILE *f = fopen(name, "rb");
    char buffer[NERSC_BUFSIZE];
    char *key, *value;
    char *dim_ok = qlua_malloc(L, S->rank * sizeof (int));
    long long volume;
    int i;
    int f_format = ntNONE;
    int f_fp = 0;
    int f_cs_p = 0;
    uint32_t f_checksum = 0;
    uint32_t d_checksum = 0;
    GaugeReader read_matrix = NULL;
    RealReader read_real = NULL;
    int site_size = 0;
    int big_endian = is_bigendian();
    double uni_eps = 0;
    QLA_D3_ColorMatrix mone;
    QLA_D_Complex cone;
    double max_eps = 0.0;
    char *site_buf = NULL;
    QLA_D3_ColorMatrix *CM_th = NULL;

    QLA_c_eq_r_plus_ir(cone, 1.0, 0.0);
    QLA_D3_M_eq_c(&mone, &cone);

    if (f == 0) {
        status = "file open error";
        goto clearerr_1;
    }

    for (i = 0; i < S->rank; i++)
        dim_ok[i] = 0;

    /* parse the header */
    if ((status == NULL) && 
        ((nersc_gethdr(f, buffer, &key, &value) != 1) ||
         (strcmp(key, "BEGIN_HEADER") != 0)))
        status = "missing header";

    while (status == NULL) {
        switch (nersc_gethdr(f, buffer, &key, &value)) {
        case 1:
            if (strcmp(key, "END_HEADER") != 0)
                status = "missing end of header";
            goto eoh;
        case 2:
            lua_pushstring(L, value);
            lua_setfield(L, -2, key);
            
            if (strcmp(key, "DATATYPE") == 0) {
                f_format = decode_hdr(L, value, f_format, ntNONE, nFMTs,
                                      "bad or conflicting DATATYPE", &status);
            } else if (strcmp(key, "FLOATING_POINT") == 0) {
                f_fp = decode_hdr(L, value, f_fp, 0, nFPs,
                                  "bad or conflicting FLOATING_POINT", &status);
            } else if (strcmp(key, "CHECKSUM") == 0) {
                if (f_cs_p)
                    status = "multiple CHECKSUMs";
                if ((status == NULL) && 
                    (sscanf(value, "%x", &f_checksum) != 1))
                    status = "illformed CHECKSUM";
                f_cs_p = 1;
            } else if (sscanf(key, "DIMENSION_%d", &i) == 1) {
                int di;
                if ((i < 1) || (i > S->rank))
                    status = "DIMENSION out of range";
                if ((status == NULL) &&
                    ((sscanf(value, "%d", &di) != 1) ||
                     (S->dim[i - 1] != di)))
                    status = "DIMENSION mismatch";
                dim_ok[i - 1] = 1;
            } else if (sscanf(key, "BOUNDARY_%d", &i) == 1) {
                if ((i < 1) || (i > S->rank))
                    status = "BOUNDARY out of range";
                if ((status == NULL) &&
                    (strcmp(value, "PERIODIC") != 0))
                    status = "bad BOUNDARY value";
            } else if ((strcmp(key, "PLAQUETTE") == 0) ||
                       (strcmp(key, "LINK_TRACE") == 0)) {
                double v;
                if (sscanf(value, "%lg", &v) != 1)
                    status = "unexpected value";
            }
            break;
        default:
            status = "illformed header";
        }
    }
eoh:
    /* get defaults from the call */
    f_format = default_enum(L, "DATATYPE", f_format, nFMTs);
    f_fp = default_enum(L, "FLOATING_POINT", f_fp, nFPs);
    uni_eps = default_double(L, ukey, 0.0);

    switch (f_format) {
    case nt4D_3x3:
        read_matrix = read_3x3;
        site_size = S->rank * 3 * 3 * 2;
        break;
    case nt4D_3x2:
        read_matrix = read_3x2;
        site_size = S->rank * 3 * (3 - 1) * 2;
        break;
    default:
        if (status == NULL)
            status = "unsupported data format";
    }
    int big_endian_data = 1;    /* default value */
    switch (f_fp) {
    case 4:
        read_real = read_float;
        site_size *= 4;
        big_endian_data = 1;
        if (uni_eps == 0) uni_eps = 1e-6;
        break;
    case 8:
        read_real = read_double;
        site_size *= 8;
        big_endian_data = 1;
        if (uni_eps == 0) uni_eps = 1e-12;
        break;
    case 16:
        read_real = read_float;
        site_size *= 4;
        big_endian_data = 0;
        if (uni_eps == 0) uni_eps = 1e-6;
        break;
    case 32:
        read_real = read_double;
        site_size *= 8;
        big_endian_data = 0;
        if (uni_eps == 0) uni_eps = 1e-12;
        break;

    default:
        if (status == NULL)
            status = "bad floating point size";
    }
    for (i = 0; i < S->rank; i++) {
        if ((!dim_ok[i]) && (status == NULL)) {
            status = "missing DIMENSION spec";
        }
    }
    if ((f_cs_p == 0) && (status == NULL))
        status = "missing CHECKSUM";

    /* Read the data and send it to the target host */
    for (volume = 1, i = 0; i < S->rank; i++)
        volume *= S->dim[i];

    /* read every site in order on the master
     * compute the checksum and send it to the target node
     */
#define SITE_BLOCK  (1024*128)
    site_buf = qlua_malloc(L, SITE_BLOCK * site_size);
    CM_th = qlua_malloc(L, SITE_BLOCK * S->rank * sizeof(CM_th[0]));
    if (NULL == site_buf || NULL == CM_th) {
        status = "not enough memory";
        goto clearerr_1;
    }

    /* Go through all sites */
    for (long long site0 = 0; site0 < volume; site0 += SITE_BLOCK) {
        int read_sites = (SITE_BLOCK <= volume - site0
                ? SITE_BLOCK : volume - site0);
        if ((status == NULL) && (fread(site_buf, read_sites * site_size, 1, f) != 1))
            status = "file read error";

        /* swap bytes if necessary */
        if ((big_endian && !big_endian_data) 
                || (!big_endian && big_endian_data)) {
            assert(4 == f_fp || 8 == f_fp);
            swap_endian(site_buf, f_fp, read_sites * site_size / f_fp);
        }
        
        /* collect the checksum */
        for (i = 0; i < read_sites * site_size; i += sizeof (uint32_t))
            d_checksum += *(uint32_t *)(site_buf + i);

        /* place ColorMatrix in U where it belongs */
#pragma omp parallel for
        for (int i_th = 0 ; i_th < read_sites ; i_th++) {
            long long site = site0 + i_th;
            QLA_D3_ColorMatrix *CM = CM_th + i_th * S->rank;
            int coord[QLUA_MAX_LATTICE_RANK];
            int s_node;
            nersc_site2coord(coord, site, S->rank, S->dim);
            s_node = QDP_node_number_L(S->lat, coord);

            if (s_node == QDP_this_node) {
                /* convert to the ColorMatrix */
                if (read_matrix != NULL && read_real != NULL) 
                    read_matrix(L, S, CM, S->rank, site_buf + i_th * site_size, 
                            site_size, read_real);
                // /* check unitarity */
                // {
                //     int d, a, b;
                //     QLA_D3_ColorMatrix UxU;

                //     for (d = 0; d < S->rank; d++) {
                //         double er, ei;

                //         /* multiplcation order helps to detect reconstruction bugs */
                //         QLA_D3_M_eq_M_times_Ma(&UxU, &CM[d], &CM[d]);
                //         QLA_D3_M_meq_M(&UxU, &mone);
                //         for (a = 0; a < 3; a++) {
                //             for (b = 0; b < 3; b++) {
                //                 QLA_Complex z;

                //                 QLA_D3_C_eq_elem_M(&z, &UxU, a, b);
                //                 er = fabs(QLA_real(z));
                //                 ei = fabs(QLA_imag(z));
                //                 if (((er > uni_eps) || (ei > uni_eps)) && (status == NULL))
                //                     status = "unitarity violation";
                //                 if (er > max_eps)
                //                     max_eps = er;
                //                 if (ei > max_eps)
                //                     max_eps = ei;
                //             }
                //         }
                //     }
                // }

                /* check unitarity modified by Jinchen on 2024-3-17 */
                {
                    int d, a, b;
                    QLA_D3_ColorMatrix UxU;
                    double max_deviation = 0.0; // new variable

                    for (d = 0; d < S->rank; d++) {
                        double er, ei;

                        /* multiplcation order helps to detect reconstruction bugs */
                        QLA_D3_M_eq_M_times_Ma(&UxU, &CM[d], &CM[d]);
                        QLA_D3_M_meq_M(&UxU, &mone);
                        for (a = 0; a < 3; a++) {
                            for (b = 0; b < 3; b++) {
                                QLA_Complex z;

                                QLA_D3_C_eq_elem_M(&z, &UxU, a, b);
                                er = fabs(QLA_real(z));
                                ei = fabs(QLA_imag(z));
                                if (er > max_eps)
                                    max_eps = er;
                                if (ei > max_eps)
                                    max_eps = ei;
                                // renew the max deviation
                                if (max_deviation < er) max_deviation = er;
                                if (max_deviation < ei) max_deviation = ei;
                            }
                        }
                    }
                    // check and print out the deviation, instead of stop
                    if (max_deviation > uni_eps) {
                        printf("Unitarity violation detected. Max deviation: %e\n", max_deviation);
                    }
                }

                long long idx = QDP_index_L(S->lat, coord);
                for (int d = 0; d < S->rank; d++)
                    QLA_D3_M_eq_M(&U[d][idx], &CM[d]);
            } 
        }
    }
clearerr_1:
    qlua_free_not_null(L, dim_ok);
    qlua_free_not_null(L, site_buf);
    qlua_free_not_null(L, CM_th);
        
    if (f)
      fclose(f);

    /* check the checksum */
    if ((status == NULL) && (f_checksum != d_checksum))
        status = "checksum mismatch";

#if 0
    /* broadcast the size of status to everyone */
    int status_len = (status != NULL)? (strlen(status) + 1): 0;
    QMP_sum_int(&status_len);
    /* if status is not NULL, broadcast to to everyone and call lua error */
    if (status_len != 0) {
        QMP_broadcast(status, status_len);
        return nersc_error(L, status);
    }
#else
    if (NULL != status)
        return luaL_error(L, "nersc_read_parallel: %s", status);
#endif
    /* convert max_eps to a string and store it (L, -2, ukey) */
    snprintf(buffer, sizeof (buffer) - 1, "%.10e", max_eps);
    lua_pushstring(L, buffer);
    lua_setfield(L, -2, ukey);
    /* normalize key/value pairs */
    normalize_kv(L, S, -1);
    return 2;
}

static int
q_nersc_read(lua_State *L)
{
    mLattice *S = qlua_checkLattice(L, 1);
    const char *name = luaL_checkstring(L, 2);
    default_setup(L);   /* creates or reuses a table? there is another one below */
    QDP_D3_ColorMatrix **M = qlua_malloc(L, S->rank * sizeof (QDP_D3_ColorMatrix *));
    QLA_D3_ColorMatrix **U = qlua_malloc(L, S->rank * sizeof (QLA_D3_ColorMatrix *));
    int status;
    int i;
    int read_parallel = 1;
    int oidx = 3;
    if (qlua_checkopt_paramtable(L, oidx)) {
        read_parallel = qlua_tabkey_boolopt(L, oidx, "parallel", 1);

    }
    /**/DEBUG_L1("parallel=%d\n", read_parallel);
    
    lua_createtable(L, S->rank, 0);
    CALL_QDP(L);
    for (i = 0; i < S->rank; i++) {
        M[i] = qlua_newLatColMat3(L, 1, 3)->ptr;
        U[i] = QDP_D3_expose_M(M[i]);
        lua_rawseti(L, -2, i + 1);
    }
    lua_newtable(L);
    if (read_parallel) {
        status = nersc_read_parallel(L, S, U, name);
    } else {
        if (QDP_this_node == qlua_master_node) {
            status = nersc_read_master(L, S, U, name);
        } else {
            status = nersc_read_slave(L, S, U, name);
        }
    }
    CALL_QDP(L);
    for (i = 0; i < S->rank; i++) {
        QDP_D3_reset_M(M[i]);
    }
    qlua_free(L, U);
    qlua_free(L, M);
    
    return status;
}

static const char *
nersc_write_master(
        lua_State *L, mLattice *S, QLA_D3_ColorMatrix **gauge_u_qla, 
        FILE *f, int site_size, int f_fp, int big_endian_data,
        RealWriter write_real, GaugeWriter write_matr, char *site_buf)
{
    const char *status = NULL;

    int coord[QLUA_MAX_LATTICE_RANK];
    QLA_D3_ColorMatrix CM[QLUA_MAX_LATTICE_RANK];
    long long volume, 
              site;
    int s_node, idx, d;
    int big_endian = is_bigendian();

    assert(NULL != f);
    
    /* get data from all nodes in nersc order */
    /* TODO group adjacent read/writes */
    QMP_msgmem_t mm = QMP_declare_msgmem(&CM[0], S->rank * sizeof(CM[0]));
    volume = QDP_volume_L(S->lat);
    for (site = 0; site < volume; site++) {
        nersc_site2coord(coord, site, S->rank, S->dim);
        s_node = QDP_node_number_L(S->lat, coord);
        if (s_node == QDP_this_node) {
            idx = QDP_index_L(S->lat, coord);
            for (d = 0; d < S->rank; d++)
                QLA_D3_M_eq_M(&CM[d], &gauge_u_qla[d][idx]);
        } else {
            QMP_msghandle_t mh = QMP_declare_receive_from(mm, s_node, 0);
            QMP_start(mh);
            QMP_wait(mh);
            QMP_free_msghandle(mh);
        }
        /* convert to file format */
        write_matr(L, S, site_buf, CM, write_real);
        /* swap bytes if necessary */
        if ((big_endian && !big_endian_data) 
                || (!big_endian && big_endian_data))
            swap_endian(site_buf, f_fp, site_size / f_fp);
        if (fwrite(site_buf, site_size, 1, f) != 1)
            status = "file read error";
    }
    QMP_free_msgmem(mm);
        
    return status;
}
static const char *
nersc_write_slave(lua_State *L, mLattice *S, QLA_D3_ColorMatrix **gauge_u_qla)
{
    const char *status = NULL;
    long long volume, 
              site;
    int s_node, idx, d;
    int coord[QLUA_MAX_LATTICE_RANK];
    QLA_D3_ColorMatrix CM[QLUA_MAX_LATTICE_RANK];

    /* get all data to qlua_master_node in nersc order */
    /* TODO group adjacent read/writes */
    QMP_msgmem_t mm = QMP_declare_msgmem(&CM[0], S->rank * sizeof (CM[0]));
    volume = QDP_volume_L(S->lat);
    for (site = 0; site < volume; site++) {
        nersc_site2coord(coord, site, S->rank, S->dim);
        s_node = QDP_node_number_L(S->lat, coord);
        if (s_node == QDP_this_node) {
            idx = QDP_index_L(S->lat, coord);
            for (d = 0; d < S->rank; d++)
                QLA_D3_M_eq_M(&CM[d], &gauge_u_qla[d][idx]);
            QMP_msghandle_t mh = QMP_declare_send_to(mm, qlua_master_node, 0);
            QMP_start(mh);
            QMP_wait(mh);
            QMP_free_msghandle(mh);
        }
    }
    QMP_free_msgmem(mm);
    
    return status;
}

/* qcd.nesrc.write_gauge(filename, gauge[], opt{})*/
static int
q_nersc_write(lua_State *L)
{
    CHK_ERR_DECL;
    const char *status = NULL;
    mLattice *S = NULL;
    const char *fname = luaL_checkstring(L, 1);
    FILE *f = NULL;
    char *site_buf = NULL;
    QDP_D_Complex *lat_c = NULL;
    QDP_D3_ColorMatrix *aux1 = NULL, 
                       *aux2 = NULL;
    QDP_D3_ColorMatrix *gauge_u[QLUA_MAX_LATTICE_RANK];
    QLA_D3_ColorMatrix *gauge_u_qla[QLUA_MAX_LATTICE_RANK];
    QLA_D3_ColorMatrix CM[QLUA_MAX_LATTICE_RANK];
    int gauge_len,
        gauge_ndim;
    double gauge_linktrace = 0.,
           gauge_plaquette = 0.;
    uint32_t nersc_cksum = 0,
             nersc_cksum_local = 0;
    QDP_Shift *neighbor;
    QLA_D_Complex scal_c;
    assert(0 <= qlua_master_node) ; /* make sure there is master node */
    int is_master_node = (QDP_this_node == qlua_master_node);
    RealWriter write_real = NULL;
    GaugeWriter write_matr = NULL;
    
    /* FIXME find ``parse_gauge'' function and replace */
    qlua_checktable(L, 2, "gauge field expected");
    gauge_len = lua_objlen(L, 2);
    CHK_ERR(gauge_len <= 0, "empty table: gauge field expected");
    CHK_ERR(QLUA_MAX_LATTICE_RANK < gauge_len, "table too large: gauge field expected");
    for (int mu = 0; mu < gauge_len ; mu++) {
        lua_rawgeti(L, 2, 1 + mu);
        if (NULL == S) { 
            S = qlua_ObjLattice(L, -1);
            lua_pop(L, 1);
        }
        gauge_u[mu] = qlua_checkLatColMat3(L, -1, S, 3)->ptr;
        lua_pop(L, 1);
    }
    gauge_ndim = S->rank ;
    CHK_ERR(gauge_ndim != gauge_len, "gauge field[2]: table size != lattice rank");

    lat_c   = QDP_D_create_C_L(S->lat);
    aux1    = QDP_D3_create_M_L(S->lat);
    aux2    = QDP_D3_create_M_L(S->lat);
    CHK_ERR(NULL == lat_c || NULL == aux1 || NULL == aux2, "not enough memory");
    neighbor = QDP_neighbor_L(S->lat);
    QDP_Subset qall = QDP_all_L(S->lat);

    long long volume = QDP_volume_L(S->lat);
    gauge_linktrace = 0;
    gauge_plaquette = 0;
    for (int mu = 0 ; mu < gauge_ndim ; mu++) {
        QDP_D3_C_eq_trace_M(lat_c, gauge_u[mu], qall);
        QDP_D_c_eq_sum_C(&scal_c, lat_c, qall);
        gauge_linktrace += QLA_real(scal_c);
        for (int nu = 0 ; nu < mu ; nu++) {
            QDP_D3_M_eq_M_times_sM(aux1, gauge_u[mu], gauge_u[nu], neighbor[mu], QDP_forward, qall);
            QDP_D3_M_eq_M_times_sM(aux2, gauge_u[nu], gauge_u[mu], neighbor[nu], QDP_forward, qall);
            QDP_D3_C_eq_M_dot_M(lat_c, aux2, aux1, qall);
            QDP_c_eq_sum_C(&scal_c, lat_c, qall);
            gauge_plaquette += QLA_real(scal_c);
        }
    }
    gauge_linktrace *= 1. / (3. * volume * S->rank);
    gauge_plaquette *= 2. / (3. * volume * S->rank * (S->rank - 1));
    for (int mu = 0 ; mu < gauge_ndim ; mu++)
        gauge_u_qla[mu] = QDP_D3_expose_M(gauge_u[mu]);

    const char  *str_NONE   = "NONE";
    const char  *hdr_HDR_VERSION        = "1.0",
                *hdr_STORAGE_FORMAT     = "1.0",
                *hdr_DATATYPE           = "4D_SU3_GAUGE",
                *hdr_ENSEMBLE_ID        = str_NONE,
                *hdr_ENSEMBLE_LABEL     = str_NONE,
                *hdr_SEQUENCE_NUMBER    = "0",
                *hdr_CREATOR            = str_NONE,
                *hdr_CREATOR_HARDWARE   = str_NONE,
                *hdr_CREATION_DATE      = str_NONE,
                *hdr_ARCHIVE_DATE       = str_NONE,
                *hdr_FLOATING_POINT     = "IEEE64BIG";
    int write_parallel = 0;

    const int oidx = 3;       /* tabopt index */
    if (qlua_checkopt_paramtable(L, oidx)) {
        write_parallel          = qlua_tabkey_boolopt(L, oidx, "parallel", 0);
        hdr_HDR_VERSION         = qlua_tabkey_stringopt(L, oidx, "HDR_VERSION",     "1.0");
        hdr_STORAGE_FORMAT      = qlua_tabkey_stringopt(L, oidx, "STORAGE_FORMAT",  "1.0");
        hdr_ENSEMBLE_ID         = qlua_tabkey_stringopt(L, oidx, "ENSEMBLE_ID",     str_NONE);
        hdr_ENSEMBLE_LABEL      = qlua_tabkey_stringopt(L, oidx, "ENSEMBLE_LABEL",  str_NONE);
        hdr_SEQUENCE_NUMBER     = qlua_tabkey_stringopt(L, oidx, "SEQUENCE_NUMBER", str_NONE);
        hdr_CREATOR             = qlua_tabkey_stringopt(L, oidx, "CREATOR",         "Qlua");
        hdr_CREATOR_HARDWARE    = qlua_tabkey_stringopt(L, oidx, "CREATOR_HARDWARE", str_NONE);
        hdr_CREATION_DATE       = qlua_tabkey_stringopt(L, oidx, "CREATION_DATE",   str_NONE);
        hdr_ARCHIVE_DATE        = qlua_tabkey_stringopt(L, oidx, "ARCHIVE_DATE",    str_NONE);
        hdr_FLOATING_POINT      = qlua_tabkey_stringopt(L, oidx, "FLOATING_POINT",  "IEEE64BIG");
        hdr_DATATYPE            = qlua_tabkey_stringopt(L, oidx, "DATATYPE",        "4D_SU3_GAUGE");
    }

    int site_size = 0;
    if        (!strcmp(hdr_DATATYPE, "4D_SU3_GAUGE_3x3")) {
        write_matr = write_3x3;
        site_size = S->rank * 3 * 3 * 2;
    } else if (!strcmp(hdr_DATATYPE, "4D_SU3_GAUGE")) {
        write_matr = write_3x2;
        site_size = S->rank * 3 * (3 - 1) * 2;
    } else 
        CHK_ERR_V(1, "bad DATATYPE '%s'", hdr_DATATYPE);

    int big_endian_data = 1;    /* default value */
    int word_size = 0;
    if        (!strcmp(hdr_FLOATING_POINT, "IEEE32") 
            || !strcmp(hdr_FLOATING_POINT, "IEEE32BIG")) {
        word_size       = 4;
        write_real      = write_float;
        big_endian_data = 1;
    } else if (!strcmp(hdr_FLOATING_POINT, "IEEE32LITTLE")) {
        word_size       = 4;
        write_real      = write_float;
        big_endian_data = 0;
    } else if (!strcmp(hdr_FLOATING_POINT, "IEEE64BIG")) {
        word_size       = 8;
        write_real      = write_double;
        big_endian_data = 1;
    } else if (!strcmp(hdr_FLOATING_POINT, "IEEE64LITTLE")) {
        word_size       = 8;
        write_real      = write_double;
        big_endian_data = 0;
    } else 
        CHK_ERR_V(1, "bad FLOATING_POINT '%s'", hdr_FLOATING_POINT);
    site_size *= word_size;
    site_buf= qlua_malloc(L, site_size);
    CHK_ERR(NULL == site_buf, "not enough memory");

    /* compute checksum */
    nersc_cksum_local = 0;
    int loc_vol = QDP_sites_on_node_L(S->lat);
    for (int idx = 0; idx < loc_vol; idx++) {
        for (int d = 0; d < S->rank; d++)
            QLA_D3_M_eq_M(&CM[d], &gauge_u_qla[d][idx]);
        write_matr(L, S, site_buf, CM, write_real);
        for (int i = 0; i < site_size; i += sizeof (uint32_t))
            nersc_cksum_local += *(uint32_t *)(site_buf + i);
    }
    MPI_Allreduce(&nersc_cksum_local, &nersc_cksum, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

    /* write header */
    if (is_master_node) {
        f = fopen(fname, "w");
        CHK_ERR_V(NULL == f, "%s: %s", fname, strerror(errno));

        fprintf(f, "BEGIN_HEADER\n");
        fprintf(f, "HDR_VERSION = %s\n", hdr_HDR_VERSION);
        fprintf(f, "STORAGE_FORMAT = %s\n", hdr_STORAGE_FORMAT);
        fprintf(f, "DATATYPE = %s\n", hdr_DATATYPE);
        fprintf(f, "LINK_TRACE = %.12f\n", gauge_linktrace);       /* FIXME check normalization */
        fprintf(f, "PLAQUETTE = %.12f\n", gauge_plaquette);        /* FIXME check normalization */
        for (int mu = 0 ; mu < gauge_ndim ; mu++)
            fprintf(f, "DIMENSION_%d = %d\n", 1 + mu, S->dim[mu]);
        for (int mu = 0 ; mu < gauge_ndim ; mu++)
            fprintf(f, "BOUNDARY_%d = %s\n", 1 + mu, "PERIODIC"); /* FIXME support other boundary types? */
        fprintf(f, "ENSEMBLE_ID = %s\n", hdr_ENSEMBLE_ID);
        fprintf(f, "ENSEMBLE_LABEL = %s\n", hdr_ENSEMBLE_LABEL);
        fprintf(f, "SEQUENCE_NUMBER = %s\n", hdr_SEQUENCE_NUMBER);
        fprintf(f, "CREATOR = %s\n", hdr_CREATOR);
        fprintf(f, "CREATOR_HARDWARE = %s\n", hdr_CREATOR_HARDWARE);
        fprintf(f, "CREATION_DATE = %s\n", hdr_CREATION_DATE);
        fprintf(f, "ARCHIVE_DATE = %s\n", hdr_ARCHIVE_DATE);
        fprintf(f, "FLOATING_POINT = %s\n", hdr_FLOATING_POINT);
        fprintf(f, "CHECKSUM = %x\n", nersc_cksum);
        fprintf(f, "END_HEADER\n");
    }

    /* write data */
    if (write_parallel) {
        CHK_ERR(write_parallel, "not implemented");
//        status = nersc_write_parallel(L, S, gauge_u_qla, name, write_real, write_gauge);
    } else {
        if (is_master_node) {
            status = nersc_write_master(L, S, gauge_u_qla, 
                    f, site_size, word_size, big_endian_data, 
                    write_real, write_matr, site_buf);
        } else {
            status = nersc_write_slave(L, S, gauge_u_qla);
        }
    }

    if (is_master_node && NULL != f) {
        fflush(f);
        fclose(f);
    }
    
    for (int mu = 0 ; mu < gauge_ndim ; mu++)
        QDP_D3_reset_M(gauge_u[mu]);

CHK_ERR_clear_0:

    /* cleanup */
    if (NULL != lat_c) QDP_D_destroy_C(lat_c);
    if (NULL != aux1) QDP_D3_destroy_M(aux1);
    if (NULL != aux2) QDP_D3_destroy_M(aux2);
    if (NULL != site_buf) qlua_free(L, site_buf);

    /* TODO fix duplicated error detection */
    CHK_ERR_LUA_REPORT(L);
    if (NULL != status)
        luaL_error(L, status);

    return 0;
}


static const struct luaL_Reg fNERSC[] = {
    { "read_gauge",     q_nersc_read },
    { "write_gauge",    q_nersc_write },
    { NULL,             NULL         }
};

int
init_nersc_io(lua_State *L)
{
    lua_getglobal(L, qcdlib);
    lua_newtable(L);
    luaL_register(L, NULL, fNERSC);
    lua_setfield(L, -2, nersc_io);
    lua_pop(L, 1);

    return 0;
}
#else /* USE_Nc3 */
int
init_nersc_io(lua_State *L)
{
    return 0;
}
#endif /* USE_Nc3 == 0 */

void
fini_nersc_io(void)
{
}
