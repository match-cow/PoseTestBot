package application;

/**
 * Read-only pose-stream control shared by PoseTestBot robot applications and
 * the automatic cyclic background task.
 *
 * This interface deliberately contains no robot-motion operation. KUKA
 * background tasks may query robot data, but must not command motion or alter
 * motion-related parameters.
 */
public interface PoseTestBotPoseStreamFunction {
	void configure(String receiverIp, int receiverPort, String runId,
			String referenceFramePath);

	void startMotion(String motionName);

	long stopMotion();

	boolean sendCurrentPose(String motionName);

	void finishCapture();

	int getTargetPeriodMs();

	long getSentPoseCount();

	long getSendFailureCount();

	long getFatalFailureCount();

	long getMaximumPoseDeltaNs();

	long getMaximumPoseQueryDurationNs();

	String getLastError();
}
