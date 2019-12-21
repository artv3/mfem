#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "additive_schwarzp.hpp"
#include "../as/schwarzp.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../../data/star.mesh";
   int order = 1;
   int ref_levels = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-ref", "--ref_levels",
                  "Number of uniform h-refinements");                              
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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


   // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   Mesh * mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, 1, 1, false);
   int dim = mesh->Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   mesh->ReorientTetMesh();

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   // FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   ParGridFunction x(fespace);
   x = 0.0;

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   // a->AddDomainIntegrator(new CurlCurlIntegrator(one));
   a->Assemble();

   HypreParMatrix A;
   // OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   ParAddSchwarz *prec2 = new ParAddSchwarz(a);
   prec2->SetOperator(A);
   prec2->SetNumSmoothSteps(1);



   int maxit = 2000;
   double rtol = 1e-8;
   double atol = 1e-8;
   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetPrintLevel(1);
   pcg.SetMaxIter(maxit);
   pcg.SetRelTol(rtol);
   pcg.SetAbsTol(atol);
   pcg.SetOperator(A);
   pcg.SetPreconditioner(*prec2); 
   X = 0.0;
   pcg.Mult(B, X);

   a->RecoverFEMSolution(X, *b, x);

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *pmesh  << "keys n/n" << flush;
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