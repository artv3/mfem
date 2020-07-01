//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               mpirun -np 4 ex1p -pa -d cuda
//               mpirun -np 4 ex1p -pa -d occa-cuda
//               mpirun -np 4 ex1p -pa -d raja-omp
//               mpirun -np 4 ex1p -pa -d ceed-cpu
//               mpirun -np 4 ex1p -pa -d ceed-cuda
//               mpirun -np 4 ex1p -m ../data/beam-tet.mesh -pa -d ceed-cpu
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void myGetLocalA(const HypreParMatrix &in_A,
                 Array<HYPRE_Int> &I, Array<int64_t> &J, Array<double> &Data)
{

   mfem::SparseMatrix Diag, Offd;
   HYPRE_Int* cmap; //column map

   in_A.GetDiag(Diag); Diag.SortColumnIndices();
   in_A.GetOffd(Offd, cmap); Offd.SortColumnIndices();

   //Number of rows in this partition
   int row_len = std::abs(in_A.RowPart()[1] -
                          in_A.RowPart()[0]); //end of row partition

   //Note Amgx requires 64 bit integers for column array
   //So we promote in this routine
   int *DiagI = Diag.GetI();
   int *DiagJ = Diag.GetJ();
   double *DiagA = Diag.GetData();

   int *OffI = Offd.GetI();
   int *OffJ = Offd.GetJ();
   double *OffA = Offd.GetData();

   I.SetSize(row_len+1);

   //Enumerate the local rows [0, num rows in proc)
   I[0]=0;
   for (int i=0; i<row_len; i++)
   {
      I[i+1] = I[i] + (DiagI[i+1] - DiagI[i]) + (OffI[i+1] - OffI[i]);
   }

   const HYPRE_Int *colPart = in_A.ColPart();
   J.SetSize(I[row_len]); //J = -777; 
   Data.SetSize(I[row_len]); //Data = -777;

   int cstart = colPart[0];

   int k    = 0;
   for (int i=0; i<row_len; i++)
   {

      int jo, icol;
      int ncols_o = OffI[i+1] - OffI[i];
      int ncols_d = DiagI[i+1] - DiagI[i];

      //OffDiagonal
      for (jo=0; jo<ncols_o; jo++)
      {
         icol = cmap[*OffJ];
         if (icol >= cstart) { break; }
         J[k]   = icol; OffJ++;
         Data[k++] = *OffA++;
      }

      //Diagonal matrix
      for (int j=0; j<ncols_d; j++)
      {
         J[k]   = cstart + *DiagJ++;
         Data[k++] = *DiagA++;
      }

      //OffDiagonal
      for (int j=jo; j<ncols_o; j++)
      {
         J[k]   = cmap[*OffJ++];
         Data[k++] = *OffA++;
      }
   }

}

void GatherArray(Array<double> &inArr, Array<double> &outArr,
                 int MPI_SZ)
{
   //Calculate number of elements to be collected from each process
   mfem::Array<int> Apart(MPI_SZ);
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,MPI_COMM_WORLD);
                 
   MPI_Barrier(MPI_COMM_WORLD);

   //Determine stride for process
   mfem::Array<int> Adisp(MPI_SZ);
   Adisp[0] = 0; 
   for(int i=1; i<MPI_SZ; ++i){
     Adisp[i] = Adisp[i-1] + Adisp[i];
   }

   MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_DOUBLE, 
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD); 
} 

void GatherArray(Array<int> &inArr, Array<int> &outArr,
                 int MPI_SZ)
{
   //Calculate number of elements to be collected from each process
   mfem::Array<int> Apart(MPI_SZ);
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,MPI_COMM_WORLD);
                 
   MPI_Barrier(MPI_COMM_WORLD);

   //Determine stride for process
   mfem::Array<int> Adisp(MPI_SZ);
   Adisp[0] = 0; 
   for(int i=1; i<MPI_SZ; ++i){
     Adisp[i] = Adisp[i-1] + Adisp[i];
   }

   MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_INT, 
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_INT, 0, MPI_COMM_WORLD); 
} 

void GatherArray(Array<int64_t> &inArr, Array<int64_t> &outArr,
                 int MPI_SZ)
{
   //Calculate number of elements to be collected from each process
   mfem::Array<int> Apart(MPI_SZ);
   int locAsz = inArr.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,MPI_COMM_WORLD);
                 
   MPI_Barrier(MPI_COMM_WORLD);

   //Determine stride for process
   mfem::Array<int> Adisp(MPI_SZ);
   Adisp[0] = 0; 
   for(int i=1; i<MPI_SZ; ++i){
     Adisp[i] = Adisp[i-1] + Adisp[i];
   }

   MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_INT64_T, 
               outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_INT64_T, 0, MPI_COMM_WORLD); 
   
   MPI_Barrier(MPI_COMM_WORLD);
} 

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   //const char *mesh_file = "../data/star.mesh";
   const char *mesh_file = "../data/inline-quad.mesh";

   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool amgx = true;
   const char *device_config = "cpu";
   bool visualization = true;
   const char *amgx_cfg = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&amgx_cfg, "-c","--c","AMGX solver file");
   args.AddOption(&amgx, "-amgx","--amgx","-no-amgx",
                  "--no-amgx","Use AMGX");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels = 0;
      //(int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   //AMGX
   std::string amgx_str;
   amgx_str = amgx_cfg;
   //NvidiaAMGX amgx;
   //amgx.Init(MPI_COMM_WORLD, "dDDI", amgx_str);   
   //amgx.SetA(A);   
   //X = 0.0; //set to zero
   //amgx.Solve(X, B);
   
   //Mini AMGX Parallel Example 
   //   {     
     int MPI_SZ, MPI_RANK;
     
     AMGX_matrix_handle      A_amgx;
     AMGX_vector_handle x_amgx, b_amgx;
     AMGX_solver_handle solver_amgx;
     
     AMGX_Mode amgx_mode = AMGX_mode_dDDI; 

     int ring;
     AMGX_config_handle  cfg;    
     static AMGX_resources_handle   rsrc;
     
     //Local processor 
     Array<int> loc_I;
     Array<int64_t> loc_J;
     Array<double> loc_A;
     
     MPI_Comm_size(MPI_COMM_WORLD, &MPI_SZ);
     MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);
     
     int nDevs, deviceId; 
     cudaGetDeviceCount(&nDevs);
     cudaGetDevice(&deviceId);

     printf("No of devices %d deviceId %d \n", nDevs, deviceId); 
#if 0
     AMGX_SAFE_CALL(AMGX_initialize());
     
     AMGX_SAFE_CALL(AMGX_initialize_plugins());
     
     AMGX_SAFE_CALL(AMGX_install_signal_handler());

     AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, amgx_str.c_str()));
     
     //Number of devices is set to 1
     AMGX_resources_create(&rsrc, cfg, &MPI_COMM_WORLD, 1, &deviceId);

     AMGX_vector_create(&x_amgx, rsrc, amgx_mode);
     AMGX_vector_create(&b_amgx, rsrc, amgx_mode);
     
     AMGX_matrix_create(&A_amgx, rsrc, amgx_mode);
     
     AMGX_solver_create(&solver_amgx, rsrc, amgx_mode, cfg);
     
     // obtain the default number of rings based on current configuration
     AMGX_config_get_default_number_of_rings(cfg, &ring);
#endif
     int nLocalRows; 
     int nGlobalRows = A.M();
     int globalNNZ = A.NNZ(); 

     //Step 1. 
     //Merge Diagonal and OffDiagonal into a single CSR matrix
     myGetLocalA(A, loc_I, loc_J, loc_A);


     //all_Jsz; 
     int J_allsz(0), all_NNZ(0);
     const int loc_Jz_sz = loc_J.Size();
     const int loc_A_sz = loc_A.Size();
     
     MPI_Allreduce(&loc_Jz_sz, &J_allsz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(&loc_A_sz, &all_NNZ, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
     MPI_Barrier(MPI_COMM_WORLD); 


     printf("all_Jz %d all_NNZ %d global_NNZ %d \n", J_allsz, all_NNZ, globalNNZ); 

     printf("loc_I.Size() %d \n", loc_I.Size());
     printf("nGlobalRows+MPI_SZ %d \n", nGlobalRows+MPI_SZ);
     
   //Consolidate to 1 rank. 
   Array<int> all_I(nGlobalRows+MPI_SZ);
   Array<int64_t> all_J(J_allsz); all_J = 0.0;
   Array<double> all_A(all_NNZ); 
   for(int i=0; i<all_A.Size(); ++i) {
      all_A[i] = i; 
   }


   //all_I
   printf("loc_J size %d all_Jsz %d \n", loc_J.Size(), all_J.Size());

   //all_J


   GatherArray(loc_I, all_I, MPI_SZ); 
   GatherArray(loc_J, all_J, MPI_SZ); 
   GatherArray(loc_A, all_A, MPI_SZ); 

#if 0
   mfem::Array<int> Apart(MPI_SZ);
   int locAsz = loc_A.Size();
   MPI_Allgather(&locAsz, 1, MPI_INT,
                 Apart.GetData(),1, MPI_INT,MPI_COMM_WORLD);
                 
   MPI_Barrier(MPI_COMM_WORLD);

   mfem::Array<int> Adisp(MPI_SZ);
   for(int i=0; i<MPI_SZ; ++i){
     Adisp[i] = Apart[i]*i;
   }
   
   MPI_Gatherv(loc_A.HostReadWrite(), loc_A.Size(), MPI_DOUBLE,
               all_A.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD); 

   MPI_Barrier(MPI_COMM_WORLD);
#endif

   printf("loc_A[0] = %f \n", loc_A[0]);
   printf("all_A[0] = %f \n", all_A[0]);

   #if 0
   if(MPI_RANK==0) {
     //all_I.Print();
     //printf("\n"); 
     //printf("rowPart.Size() %d \n", rowPart.Size());
     
     for(int idx=loc_I.Size(); idx<all_I.Size()-1; ++idx){
       all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
     }

     //printf("all of I \n"); 
     //all_I.Print();
     nLocalRows = A.M();
   }else{
     nLocalRows = 0;
   }        

   //Step 2.
   //Create a vector of offsets describing matrix row partitions
   mfem::Array<int64_t> rowPart(2); 
   rowPart[0] = 0; 
   rowPart[1] = A.M();

   /*
   mfem::Array<int64_t> rowPart(MPI_SZ+1); rowPart = 0.0;

   //Must be promoted to int64!  --consider typedef?
   int64_t myStart = A.GetRowStarts()[0];
   MPI_Allgather(&myStart, 1, MPI_INT64_T,
                 rowPart.GetData(),1, MPI_INT64_T
                 ,MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);

   rowPart[MPI_SZ] = A.M();
   */
   
   AMGX_distribution_handle dist;
   AMGX_distribution_create(&dist, cfg);
   AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS,
                                        rowPart.GetData());

     #endif


   //}//Mini example end

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 17. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}
