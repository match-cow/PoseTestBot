package application;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.util.concurrent.TimeUnit;

import javax.inject.Inject;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import com.kuka.roboticsAPI.deviceModel.JointPosition;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.AbstractFrame;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;
import com.kuka.roboticsAPI.persistenceModel.templateModel.InfoTemplate;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

public class PoseTestBot_Test extends RoboticsAPIApplication {
	private static final int SAMPLE_TIME_MS = 10;
	private static final int SETTLE_TIME_MS = 500;
	private static final int ROBOT_PORT = 30300;
	private static final int DEFAULT_RECEIVER_PORT = 8080;
	private static final int END_PACKET_COUNT = 3;
	private static final int END_PACKET_INTERVAL_MS = 50;

	private static final double PTP_VEL = 0.5;
	private static final double A1_MIN_RAD = Math.toRadians(-169.0);
	private static final double A1_MAX_RAD = Math.toRadians(169.0);

	@Inject
	private LBR robot;
	@Inject
	private InfoTemplate robotinfo;

	private AbstractFrame templateBaseRef;
	private JointPosition initialJointPosition;
	private String receiverIp = "172.31.1.151";
	private int receiverPort = DEFAULT_RECEIVER_PORT;

	@Override
	public void initialize() {
		robot = getContext().getDeviceFromType(LBR.class);
		robotinfo.setBase("/HRC_Hub/Template_Base");
		getLogger().info("getBase: " + robotinfo.getBase());
		getLogger().info("getTool: " + robotinfo.getTool());
		getLogger().info("getValue: " + robotinfo.getValue());
		templateBaseRef = getApplicationData().getFrame("/HRC_Hub/Template_Base");
	}

	@Override
	public void run() {
		initialJointPosition = robot.getCurrentJointPosition();
		moveToA1Min();

		while (true) {
			Double captureVelocity = waitForStartCommand();
			if (captureVelocity == null) {
				return;
			}

			getLogger().info("Starting A1 capture sweep...");
			moveToA1Min();
			sleep(SETTLE_TIME_MS);

			IMotionContainer motion = robot.moveAsync(ptp(jointTarget(A1_MAX_RAD))
					.setJointVelocityRel(captureVelocity.doubleValue()));
			transmitPose(motion, SAMPLE_TIME_MS, "circ_1");

			transmitCurrentPose("end");
			sleep(SETTLE_TIME_MS);
			moveToA1Min();
		}
	}

	private void moveToA1Min() {
		robot.move(ptp(jointTarget(A1_MIN_RAD)).setJointVelocityRel(PTP_VEL));
	}

	private JointPosition jointTarget(double a1Rad) {
		JointPosition target = new JointPosition(
				initialJointPosition.get(0),
				initialJointPosition.get(1),
				initialJointPosition.get(2),
				initialJointPosition.get(3),
				initialJointPosition.get(4),
				initialJointPosition.get(5),
				initialJointPosition.get(6));
		target.set(0, a1Rad);
		return target;
	}

	private Double waitForStartCommand() {
		while (true) {
			DatagramSocket socket = null;

			try {
				socket = new DatagramSocket(ROBOT_PORT);
				getLogger().info("Waiting for UDP message...");

				byte[] receiveData = new byte[1024];
				DatagramPacket receivePacket = new DatagramPacket(receiveData,
						receiveData.length);
				socket.receive(receivePacket);

				String jsonMessage = new String(receivePacket.getData(), 0,
						receivePacket.getLength());
				getLogger().info("Received message: " + jsonMessage);

				JSONObject jsonObject = (JSONObject) new JSONParser().parse(
						jsonMessage);

				Double startValue = startValue(jsonObject);
				if (startValue != null) {
					updateReceiverTarget(jsonObject, receivePacket);
					getLogger().info("Start value: " + startValue);
					return startValue;
				}

				if (isStopCommand(jsonObject)) {
					getLogger().info("Stop message received. Ending program.");
					return null;
				}

				getLogger().info("Waiting for message with 'start' key...");
			} catch (Exception e) {
				getLogger().info("Exception: " + e);
			} finally {
				if (socket != null) {
					socket.close();
				}
			}
		}
	}

	private Double startValue(JSONObject jsonObject) {
		Object startValue = jsonObject.get("start");
		if (startValue != null) {
			return doubleValue(startValue);
		}

		if ("start_capture".equals(jsonObject.get("command"))) {
			return doubleValue(jsonObject.get("cartesian_velocity_m_s"));
		}

		return null;
	}

	private Double doubleValue(Object value) {
		if (value == null) {
			return null;
		}

		if (value instanceof Number) {
			return Double.valueOf(((Number) value).doubleValue());
		}

		return Double.valueOf(value.toString());
	}

	private boolean isStopCommand(JSONObject jsonObject) {
		if (Boolean.TRUE.equals(jsonObject.get("stop"))) {
			return true;
		}

		Object command = jsonObject.get("command");
		return "pause_capture".equals(command)
				|| "stop_after_current_motion".equals(command)
				|| "emergency_stop".equals(command);
	}

	private void updateReceiverTarget(JSONObject jsonObject,
			DatagramPacket receivePacket) {

		Object receiverIpValue = jsonObject.get("receiver_ip");
		if (receiverIpValue != null) {
			String requestedReceiverIp = receiverIpValue.toString().trim();
			if (requestedReceiverIp.length() > 0
					&& !requestedReceiverIp.equals("0.0.0.0")
					&& !requestedReceiverIp.equals("::")) {
				receiverIp = requestedReceiverIp;
			} else {
				receiverIp = receivePacket.getAddress().getHostAddress();
			}
		} else {
			receiverIp = receivePacket.getAddress().getHostAddress();
		}

		Object receiverPortValue = jsonObject.get("receiver_port");
		if (receiverPortValue instanceof Number) {
			receiverPort = ((Number) receiverPortValue).intValue();
		} else if (receiverPortValue != null) {
			receiverPort = Integer.parseInt(receiverPortValue.toString());
		}

		getLogger().info("Pose receiver target: " + receiverIp + ":"
				+ receiverPort);
	}

	@SuppressWarnings("unchecked")
	private byte[] currentPosePayload(String motionName) {
		Frame currentPose = robot.getCurrentCartesianPosition(robot.getFlange(),
				templateBaseRef);

		JSONObject jsonObject = new JSONObject();
		jsonObject.put("motion", motionName);
		jsonObject.put("X", currentPose.getX());
		jsonObject.put("Y", currentPose.getY());
		jsonObject.put("Z", currentPose.getZ());
		jsonObject.put("A", currentPose.getAlphaRad());
		jsonObject.put("B", currentPose.getBetaRad());
		jsonObject.put("C", currentPose.getGammaRad());

		return jsonObject.toJSONString().getBytes();
	}

	private void sendCurrentPose(DatagramSocket socket, String motionName)
			throws IOException {

		byte[] data = currentPosePayload(motionName);
		DatagramPacket packet = new DatagramPacket(data, data.length,
				InetAddress.getByName(receiverIp), receiverPort);
		socket.send(packet);
	}

	private void transmitCurrentPose(String motionName) {
		DatagramSocket socket = null;

		try {
			socket = new DatagramSocket();
			for (int i = 0; i < END_PACKET_COUNT; i++) {
				sendCurrentPose(socket, motionName);
				TimeUnit.MILLISECONDS.sleep(END_PACKET_INTERVAL_MS);
			}
		} catch (SocketException e) {
		} catch (IOException e) {
		} catch (InterruptedException e) {
		} finally {
			if (socket != null) {
				socket.close();
			}
		}
	}

	private void transmitPose(IMotionContainer motion, int sampleTimeMs,
			String motionName) {

		DatagramSocket socket = null;

		try {
			socket = new DatagramSocket();

			while (!motion.isFinished()) {
				try {
					sendCurrentPose(socket, motionName);
					TimeUnit.MILLISECONDS.sleep(sampleTimeMs);
				} catch (IOException e) {
				} catch (InterruptedException e) {
				}
			}

			synchronized (motion) {
				motion.notify();
			}
		} catch (SocketException e) {
		} finally {
			if (socket != null) {
				socket.close();
			}
		}
	}

	private void sleep(int millis) {
		try {
			Thread.sleep(millis);
		} catch (InterruptedException e) {
		}
	}
}
