#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define ERR fprintf(stderr, "ERR\n");

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FTYPE float

#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024/1)
#define BF (BSIZE/32)
#define INIT_GRP (10000000)
#define INIT_LIST (-1)
//#define THRESHOLD (8*1)
//#define BH (128/1*2)
#define BW (128*2)
#define MIN_OCC (BW*3/4)
//#define MIN_OCC (BW/4)
//#define BW (
#define SBSIZE (128)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2)
#define SSTRIDE (STHRESHOLD / SPBF)

#define SIMPLE

struct v_struct {
	int row, col;
	FTYPE val;
	int grp;
};

double vari;
struct v_struct *temp_v, *gold_temp_v;
int sc, nr, nc, ne, gold_ne, npanel, mne, mne_nr;
int nr0;

int *csr_v;
int *csr_e;
FTYPE *csr_ev;
FTYPE *ocsr_ev;

//int *mcsr_v;
int *mcsr_e; // can be short type
int *mcsr_cnt;
int *mcsr_list;

int *baddr, *saddr;
int num_dense;

int *special;
int *special2;
int special_p;



int compare2(const void *a, const void *b)
{
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
        return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}


int main(int argc, char **argv)
{
        FILE *fp;
	FILE *fpo;
        int *loc;
        char buf[300];
        int nflag, sflag;
        int pre_count=0, tmp_ne;
        int i;

        srand(time(NULL));
//fprintf(stdout,"%s,",argv[1]);

        fp = fopen(argv[1], "r");
        fpo = fopen(argv[2], "w");
        fgets(buf, 300, fp);
        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
        else sflag = 0;
        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
        else if(strstr(buf, "complex") != NULL) nflag = -1;
        else nflag = 1;

#ifdef SYM
        sflag = 1;
#endif

        while(1) {
                pre_count++;
                fgets(buf, 300, fp);
                if(strstr(buf, "%") == NULL) break;
        }
        fclose(fp);

        fp = fopen(argv[1], "r");
        for(i=0;i<pre_count;i++)
                fgets(buf, 300, fp);

        fscanf(fp, "%d %d %d", &nr, &nc, &ne);
        nr0 = nr;
        //ne *= (sflag+1);
        //nr = CEIL(nr,BH)*BH;
	//npanel = CEIL(nr,BH);

        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
        gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));

        for(i=0;i<ne;i++) {
                fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
		temp_v[i].grp = INIT_GRP;
                //temp_v[i].row--; temp_v[i].col--;

                //if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
                //        fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
                //        exit(0);
               // }
                if(nflag == 0) temp_v[i].val = (FTYPE)(rand()%10);
                else if(nflag == 1) {
                        FTYPE ftemp;
                        fscanf(fp, " %f ", &ftemp);
                        temp_v[i].val = ftemp;
                } else { // complex
                        FTYPE ftemp1, ftemp2;
                        fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
                        temp_v[i].val = ftemp1;
                }

/*                if(sflag == 1) {
                        i++;
                        temp_v[i].row = temp_v[i-1].col;
                        temp_v[i].col = temp_v[i-1].row;
                        temp_v[i].val = temp_v[i-1].val;
        		temp_v[i].grp = INIT_GRP;
	        }*/
        }
	fclose(fp);
	if(sflag == 1) {
		fprintf(fpo, "%%%MatrixMarket matrix coordinate real symmetric\n");

	} else {
		fprintf(fpo, "%%%MatrixMarket matrix coordinate real general\n");
	}
	fprintf(fpo, "%d %d %d\n", nr, nc ,ne);
	for(i=0;i<ne;i++) {
		fprintf(fpo, "%d %d %d\n", temp_v[i].row, temp_v[i].col, rand()%10);
	}
	fclose(fpo);

/*        qsort(temp_v, ne, sizeof(struct v_struct), compare2);

        loc = (int *)malloc(sizeof(int)*(ne+1));

        memset(loc, 0, sizeof(int)*(ne+1));
        loc[0]=1;
        for(i=1;i<ne;i++) {
                if(temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
                        loc[i] = 0;
                else loc[i] = 1;
        }
        for(i=1;i<=ne;i++)
                loc[i] += loc[i-1];
        for(i=ne; i>=1; i--)
                loc[i] = loc[i-1];
        loc[0] = 0;

        for(i=0;i<ne;i++) {
                temp_v[loc[i]].row = temp_v[i].row;
                temp_v[loc[i]].col = temp_v[i].col;
                temp_v[loc[i]].val = temp_v[i].val;
                temp_v[loc[i]].grp = temp_v[i].grp;
        }
        ne = loc[ne];
        temp_v[ne].row = nr;

	for(i=0;i<ne;i++) {
		fprintf(fpo, "%d %d %.1f\n", temp_v[i].row+1, temp_v[i].col+1, temp_v[i].val); 
	}
	fclose(fpo);
        free(loc);*/
}


