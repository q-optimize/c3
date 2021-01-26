from qiskit.providers import JobV1 as Job
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Result


class C3Job(Job):
    """C3Job class

    Parameters
    ----------
    Job : JobV1
        Inherits JobV1 from qiskit.providers

    """

    _async = False

    def __init__(self, backend, job_id, result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self) -> None:
        """Submit a job to the simulator"""
        return

    def result(self) -> Result:
        """Return the result of the job

        Returns
        -------
        qiskit.Result
            Result of the job simulation
        """
        return self._result

    def status(self) -> JobStatus:
        """Return job status

        Returns
        -------
        qiskit.providers.JobStatus
            Status of the job
        """
        return JobStatus.DONE
